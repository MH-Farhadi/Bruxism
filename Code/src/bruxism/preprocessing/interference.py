"""Measuring mains interference, so a contaminated channel cannot pass unnoticed again.

The defect diagnosed in ``cause.md`` survived a full pipeline, a figure set and three
reproducible runs because nothing ever *measured* how much of the signal was interference.
This module supplies that measurement, and it is called from two places:

* :mod:`bruxism.data.manifest` runs it on every recording's **raw** EMG and stores the
  result, so ``bruxism-audit`` reports a contamination fraction per recording and
  :data:`~bruxism.data.quality.QualityFlag.MAINS_CONTAMINATION` fires above a declared
  threshold;
* the filter verification harness runs it on the **filtered** windows the pipeline
  actually emits, which is the number that has to fall below 5 % for the fix to be
  accepted.

The statistic is deliberately simple: the share of in-band power lying within a narrow
band of each mains multiple. The recordings here showed 85-99.8 % at rest.

**Read the floor before reading the number.** Seven harmonics x a +/-3 Hz band is 42 Hz of
a 430 Hz passband, so a perfectly white signal scores about 0.098, not 0. The cleanest
recording in this dataset scores 0.095 -- i.e. it is at the geometric floor and carries no
detectable interference at all. Two consequences:

* the statistic separates populations well (0.10 vs 0.99) but its zero point is 0.098, and
  a target expressed as "below 5 %" is only reachable by a filter that *removes* those
  bins rather than restoring them;
* a notch bank scores near 0 afterwards because it zeroes the band, while spectral
  interpolation scores near the floor because it puts the surrounding noise level back.
  The lower number is not the cleaner signal. :func:`harmonic_excess_ratio` is the
  method-fair comparison: it measures harmonic power *relative to the neighbouring bins*,
  so restoring the floor reads as 1.0 and any surviving peak reads above it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from scipy.signal import welch

__all__ = [
    "MAINS_CONTAMINATION_THRESHOLD",
    "MAINS_HALF_WIDTH_HZ",
    "MainsContamination",
    "harmonic_band_share",
    "harmonic_excess_ratio",
    "harmonic_peak_ratios",
    "measure_mains_contamination",
]

#: Half-width of the band counted as "at" a mains harmonic. Wide enough to contain the
#: Welch mainlobe of a pure tone at the segment lengths used here, narrow enough that a
#: clean channel scores a few per cent rather than a third.
MAINS_HALF_WIDTH_HZ: Final[float] = 3.0

#: Fraction of in-band power at the mains harmonics above which a channel is flagged.
#: Declared, not tuned: a clean surface-EMG channel sits near 0.02-0.05, the contaminated
#: rest recordings here sit at 0.88-0.998, and 0.30 separates those two populations by an
#: order of magnitude on either side.
MAINS_CONTAMINATION_THRESHOLD: Final[float] = 0.30

#: Welch segment length. 4096 samples at 1200 Hz resolves 0.29 Hz, so a +/-3 Hz band
#: contains a tone's mainlobe with room to spare.
_NPERSEG: Final[int] = 4096


@dataclass(frozen=True)
class MainsContamination:
    """How much of a signal's in-band power sits at multiples of the mains frequency."""

    mains_hz: float
    band_hz: tuple[float, float]
    half_width_hz: float
    #: Pooled over channels: harmonic power / total in-band power.
    fraction: float
    #: One fraction per input channel, in channel order.
    per_channel: tuple[float, ...]
    #: Harmonic frequency (as a string key, e.g. ``"180"``) -> pooled fraction.
    per_harmonic: dict[str, float]

    @property
    def is_contaminated(self) -> bool:
        return self.fraction > MAINS_CONTAMINATION_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "mains_hz": self.mains_hz,
            "band_hz": list(self.band_hz),
            "half_width_hz": self.half_width_hz,
            "fraction": self.fraction,
            "per_channel": list(self.per_channel),
            "per_harmonic": dict(self.per_harmonic),
        }


def _welch_psd(signal: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """``(frequencies, psd)`` with ``psd`` shaped ``(n_frequencies, n_channels)``."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D (time, channels) array, got {values.shape}")
    if values.shape[0] < 16:
        raise ValueError(f"need at least 16 samples to estimate a spectrum, got {values.shape[0]}")
    nperseg = int(min(_NPERSEG, values.shape[0]))
    frequencies, psd = welch(values, fs=sampling_rate, nperseg=nperseg, axis=0)
    return frequencies, np.atleast_2d(psd.T).T


def measure_mains_contamination(
    signal: np.ndarray,
    sampling_rate: float,
    *,
    mains_hz: float = 60.0,
    band_hz: tuple[float, float] = (20.0, 450.0),
    half_width_hz: float = MAINS_HALF_WIDTH_HZ,
) -> MainsContamination:
    """Fraction of in-band power within ``half_width_hz`` of each mains multiple.

    Parameters
    ----------
    signal
        ``(n_samples,)`` or ``(n_samples, n_channels)``. Time is the first axis.
    band_hz
        The band the fraction is taken over -- the surface-EMG band, not the whole
        spectrum, because power outside it is removed before the model sees anything.

    Returns
    -------
    MainsContamination
        Pooled and per-channel fractions plus a per-harmonic breakdown. A channel with no
        in-band power at all scores 0.0 rather than raising: a flat channel is a different
        defect, reported by a different flag.
    """
    frequencies, psd = _welch_psd(signal, sampling_rate)
    low, high = band_hz
    in_band = (frequencies >= low) & (frequencies <= high)
    total = psd[in_band].sum(axis=0)

    harmonic_total = np.zeros_like(total)
    per_harmonic: dict[str, float] = {}
    harmonic = mains_hz
    while harmonic <= min(high, sampling_rate / 2.0):
        if harmonic >= low:
            mask = in_band & (np.abs(frequencies - harmonic) <= half_width_hz)
            power = psd[mask].sum(axis=0)
            harmonic_total += power
            per_harmonic[f"{harmonic:g}"] = _ratio(power.sum(), total.sum())
        harmonic += mains_hz

    return MainsContamination(
        mains_hz=mains_hz,
        band_hz=(low, high),
        half_width_hz=half_width_hz,
        fraction=_ratio(harmonic_total.sum(), total.sum()),
        per_channel=tuple(
            _ratio(float(h), float(t)) for h, t in zip(harmonic_total, total, strict=True)
        ),
        per_harmonic=per_harmonic,
    )


def harmonic_band_share(
    mains_hz: float = 60.0,
    *,
    band_hz: tuple[float, float] = (20.0, 450.0),
    half_width_hz: float = MAINS_HALF_WIDTH_HZ,
    sampling_rate: float = 1200.0,
) -> float:
    """Fraction of the band occupied by the harmonic windows -- the statistic's zero point.

    A white signal scores this value on :func:`measure_mains_contamination`; anything at or
    below it carries no measurable interference.
    """
    low, high = band_hz
    ceiling = min(high, sampling_rate / 2.0)
    covered = 0.0
    harmonic = mains_hz
    while harmonic <= ceiling:
        if harmonic >= low:
            covered += min(harmonic + half_width_hz, high) - max(harmonic - half_width_hz, low)
        harmonic += mains_hz
    return _ratio(covered, high - low)


def harmonic_excess_ratio(
    signal: np.ndarray,
    sampling_rate: float,
    *,
    mains_hz: float = 60.0,
    band_hz: tuple[float, float] = (20.0, 450.0),
    half_width_hz: float = MAINS_HALF_WIDTH_HZ,
    floor_half_width_hz: float = 25.0,
) -> dict[str, float]:
    """Mean PSD at each harmonic divided by the mean PSD of its neighbouring bins.

    The method-fair contamination measure. 1.0 means the harmonic bins are
    indistinguishable from their surroundings; 0.0 means they were emptied; anything above
    ~2 means interference survives. Unlike :func:`measure_mains_contamination` this does
    not reward a filter for deleting bandwidth.

    Returns a mapping with one entry per harmonic plus ``"max"`` over all of them.
    """
    frequencies, psd = _welch_psd(signal, sampling_rate)
    pooled = psd.mean(axis=1)
    low, high = band_hz
    ratios: dict[str, float] = {}
    harmonic = mains_hz
    while harmonic <= min(high, sampling_rate / 2.0):
        if harmonic >= low:
            offset = np.abs(frequencies - harmonic)
            peak = offset <= half_width_hz
            floor = (
                (offset > half_width_hz)
                & (offset <= floor_half_width_hz)
                & (frequencies >= low)
                & (frequencies <= high)
            )
            if peak.any() and floor.any():
                reference = float(np.median(pooled[floor]))
                ratios[f"{harmonic:g}"] = (
                    float(pooled[peak].mean() / reference) if reference > 0 else 0.0
                )
        harmonic += mains_hz
    ratios["max"] = max(ratios.values()) if ratios else 0.0
    return ratios


def harmonic_peak_ratios(
    signal: np.ndarray,
    sampling_rate: float,
    *,
    frequencies_hz: Sequence[float],
    half_width_hz: float = MAINS_HALF_WIDTH_HZ,
    floor_half_width_hz: float = 25.0,
) -> dict[str, float]:
    """Peak-to-local-noise-floor ratio at each named frequency, per :file:`cause.md` §1.

    The floor is the median PSD of the bins flanking the peak band, so a broadband rise
    does not read as a peak. Returned as a plain mapping for direct serialisation.
    """
    frequencies, psd = _welch_psd(signal, sampling_rate)
    pooled = psd.mean(axis=1)
    ratios: dict[str, float] = {}
    for frequency in frequencies_hz:
        offset = np.abs(frequencies - frequency)
        peak = offset <= half_width_hz
        floor = (offset > half_width_hz) & (offset <= floor_half_width_hz)
        if not peak.any() or not floor.any():
            continue
        reference = float(np.median(pooled[floor]))
        ratios[f"{frequency:g}"] = (
            float(pooled[peak].max() / reference) if reference > 0 else float("inf")
        )
    return ratios


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0
