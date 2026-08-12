"""Discrete wavelet decomposition with explicit, verified band labelling.

PyWavelets returns :func:`pywt.wavedec` coefficients in the order

``[cA_L, cD_L, cD_(L-1), ..., cD_1]``

so ``coeffs[1]`` is the **lowest**-frequency detail and ``coeffs[-1]`` the highest. The
research prototype indexed ``details[0]`` as "highest frequency" (it is the lowest) and
called ``details[2]`` "D3" (at level 4 it is D2). Both errors are corrected here by never
addressing coefficients positionally: callers name a band
(:class:`WaveletBand`) and :func:`decompose` resolves it.

Nominal band edges at sampling rate ``fs``: detail ``D_k`` spans
``[fs / 2**(k+1), fs / 2**k]`` Hz and the approximation ``A_L`` spans
``[0, fs / 2**(L+1)]`` Hz. These are the ideal half-band edges of the filter cascade; real
Daubechies filters have transition bands, so :func:`band_frequencies` documents them as
nominal.

At 1200 Hz with ``level=4``:

===========  ==========================
Band         Nominal frequency range
===========  ==========================
``D1``       300 - 600 Hz
``D2``       150 - 300 Hz
``D3``       75 - 150 Hz
``D4``       37.5 - 75 Hz
``A4``       0 - 37.5 Hz
===========  ==========================

Decomposition is done **outside** the network's ``forward()`` (see
:mod:`bruxism.data.dataset`), so no GPU tensor is round-tripped through NumPy per batch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pywt

from bruxism.utils.logging import get_logger

__all__ = [
    "BAND_PATTERN",
    "BandStopbandError",
    "WaveletConfig",
    "WaveletDecomposition",
    "WaveletError",
    "assert_bands_within_passband",
    "band_frequencies",
    "band_index",
    "band_passband_gain",
    "decompose",
    "max_useful_level",
]

logger = get_logger(__name__)

#: ``A4``, ``D1`` ... case-insensitive.
BAND_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?P<kind>[AD])(?P<level>\d+)$", re.IGNORECASE)


class WaveletError(ValueError):
    """Raised for an invalid wavelet, level or band specification.

    Feature extraction never converts a failure into a zero vector: a sample whose
    transform fails is an error, not a sample of zeros.
    """


def band_index(band: str, level: int) -> int:
    """Position of ``band`` in a :func:`pywt.wavedec` output list of depth ``level``.

    ``A<level>`` maps to 0. ``D_k`` maps to ``level - k + 1``, so ``D_level`` is index 1
    (lowest-frequency detail) and ``D1`` is index ``level`` (highest-frequency detail).

    Raises
    ------
    WaveletError
        If the band name is malformed or refers to a level the decomposition lacks.
    """
    match = BAND_PATTERN.match(band.strip())
    if match is None:
        raise WaveletError(
            f"malformed band {band!r}; expected e.g. 'A4' (approximation) or 'D2' (detail)"
        )
    kind = match["kind"].upper()
    band_level = int(match["level"])
    if kind == "A":
        if band_level != level:
            raise WaveletError(
                f"approximation band {band!r} does not exist in a level-{level} "
                f"decomposition; the only approximation is 'A{level}'"
            )
        return 0
    if not 1 <= band_level <= level:
        raise WaveletError(
            f"detail band {band!r} is outside a level-{level} decomposition (valid: D1..D{level})"
        )
    return level - band_level + 1


def band_frequencies(band: str, level: int, sampling_rate: float) -> tuple[float, float]:
    """Nominal ``(low_hz, high_hz)`` edges of ``band``.

    These are the ideal dyadic half-band edges of the decomposition cascade. Real wavelet
    filters overlap across these edges; the values are for labelling figures and tables,
    not for claiming a filter's stopband.
    """
    match = BAND_PATTERN.match(band.strip())
    if match is None:
        raise WaveletError(f"malformed band {band!r}")
    kind = match["kind"].upper()
    band_level = int(match["level"])
    band_index(band, level)  # validates against the decomposition depth
    if kind == "A":
        return 0.0, sampling_rate / 2 ** (level + 1)
    return sampling_rate / 2 ** (band_level + 1), sampling_rate / 2**band_level


class BandStopbandError(WaveletError):
    """Raised when a configured wavelet band lies inside the stopband of its own chain.

    A band the filter has already deleted still produces coefficients -- the roll-off
    residue -- and the ``BatchNorm1d`` that follows every band branch rescales them back to
    unit variance. The branch then spends its capacity amplifying whatever the filter left
    behind, which for a high-pass is drift and per-recording baseline wander: a
    recording-identity feature, not a physiological one.

    This is the wavelet analogue of the mistake ``cause.md`` documents for the filter
    itself. There, a chain was verified against its own design rather than against the
    measured spectrum of the data. Here, a band list was chosen without checking it against
    the chain that feeds it. In the microphone branch of this project both errors met: the
    chain high-passes at 20 Hz and the branch reads ``A5``, nominally 0-18.75 Hz.
    """


#: Fraction of a band's nominal width that must survive the chain for it to carry signal.
#: Well below any sensible design -- a band this attenuated is not a band, it is residue.
_MIN_BAND_PASSBAND_GAIN: Final[float] = 0.05


def band_passband_gain(
    band: str,
    level: int,
    sampling_rate: float,
    magnitude: Any,
    *,
    n_points: int = 512,
) -> float:
    """Mean power gain the chain applies across a band's nominal frequency range.

    Parameters
    ----------
    magnitude
        Callable ``f(frequencies) -> |H(f)|`` for the chain feeding this band.

    Returns
    -------
    float
        1.0 when the chain passes the whole band untouched, 0.0 when it deletes it.
    """
    low, high = band_frequencies(band, level, sampling_rate)
    # A band whose lower edge is DC still has a defined mean gain; the grid just starts at
    # 0 Hz, where a high-pass is by definition at its most attenuating.
    grid = np.linspace(low, min(high, sampling_rate / 2.0), n_points)
    response = np.asarray(magnitude(grid), dtype=np.float64)
    return float(np.mean(response**2))


def assert_bands_within_passband(
    config: WaveletConfig,
    sampling_rate: float,
    magnitude: Any,
    *,
    context: str,
    acknowledged: str | None = None,
    minimum_gain: float = _MIN_BAND_PASSBAND_GAIN,
) -> dict[str, float]:
    """Refuse a band list that the filter chain in front of it has already deleted.

    Parameters
    ----------
    magnitude
        Callable ``f(frequencies) -> |H(f)|`` for the chain feeding this branch.
    context
        What is being validated, quoted in the error -- e.g. ``"mic branch"``.
    acknowledged
        ``"<name>, <date>: <reason>"``. When supplied, an offending band is logged at
        WARNING and allowed through instead of raising. This exists for one purpose: the
        published runs of this project used ``A5`` behind a 20 Hz high-pass, and they must
        keep reproducing. A new configuration that wants the same thing has to say so in
        writing, which is the whole point.

    Returns
    -------
    dict
        ``band -> mean power gain``, for every configured band.

    Raises
    ------
    BandStopbandError
        If any band's mean power gain is below ``minimum_gain`` and ``acknowledged`` is
        not set.
    """
    gains = {
        band: band_passband_gain(band, config.level, sampling_rate, magnitude)
        for band in config.bands
    }
    offenders = {band: gain for band, gain in gains.items() if gain < minimum_gain}
    if not offenders:
        return gains

    detail = "; ".join(
        f"{band} ({band_frequencies(band, config.level, sampling_rate)[0]:g}-"
        f"{band_frequencies(band, config.level, sampling_rate)[1]:g} Hz) retains "
        f"{gain:.2%} of its power"
        for band, gain in sorted(offenders.items())
    )
    if acknowledged:
        logger.warning(
            "%s: band(s) inside the filter stopband, allowed by declaration %r -- %s",
            context,
            acknowledged,
            detail,
            extra={"context": context, "acknowledged": acknowledged, "gains": gains},
        )
        return gains
    raise BandStopbandError(
        f"{context}: the filter chain feeding this branch has already removed "
        f"{'a band' if len(offenders) == 1 else 'bands'} the branch is configured to read "
        f"-- {detail}. Such a band carries the chain's roll-off residue, which the "
        f"branch's batch normalisation then rescales to unit variance, so the branch "
        f"learns per-recording baseline wander rather than signal. Choose bands inside the "
        f"passband, widen the chain, or -- if reproducing a historical run -- pass "
        f"acknowledged='<name>, <date>: <reason>'."
    )


def max_useful_level(n_samples: int, wavelet: str) -> int:
    """Deepest decomposition level that keeps coefficients longer than the filter."""
    try:
        filter_length = pywt.Wavelet(wavelet).dec_len
    except ValueError as exc:
        raise WaveletError(f"unknown discrete wavelet {wavelet!r}: {exc}") from exc
    return int(pywt.dwt_max_level(n_samples, filter_length))


@dataclass(frozen=True)
class WaveletConfig:
    """Wavelet decomposition settings for one modality.

    Attributes
    ----------
    wavelet
        A **discrete** wavelet name (``db4``, ``coif5``, ``sym8``, ...). Continuous
        families such as ``morl`` are rejected: :func:`pywt.wavedec` requires a discrete
        wavelet, and the prototype's ``pywt.cwt(..., 'db4')`` call was invalid for the
        same reason in reverse.
    level
        Decomposition depth.
    bands
        Named bands fed to the network, in the order the branches expect. Naming them
        makes the code/paper mismatch impossible to reintroduce.
    mode
        Signal-extension mode at the boundaries.
    """

    wavelet: str = "db4"
    level: int = 4
    bands: tuple[str, ...] = ("A4", "D3", "D1")
    mode: str = "symmetric"

    def __post_init__(self) -> None:
        try:
            family = pywt.Wavelet(self.wavelet)
        except ValueError as exc:
            raise WaveletError(
                f"{self.wavelet!r} is not a discrete wavelet usable with pywt.wavedec "
                f"(continuous families such as 'morl'/'mexh' are not valid here): {exc}"
            ) from exc
        if self.level < 1:
            raise WaveletError(f"level must be >= 1, got {self.level}")
        for band in self.bands:
            band_index(band, self.level)  # raises on a band the depth cannot provide
        logger.debug(
            "wavelet config %s level=%d bands=%s (dec_len=%d)",
            self.wavelet,
            self.level,
            self.bands,
            family.dec_len,
        )

    def band_frequency_table(self, sampling_rate: float) -> dict[str, tuple[float, float]]:
        """Nominal band edges for every selected band, for figure and table captions."""
        return {band: band_frequencies(band, self.level, sampling_rate) for band in self.bands}

    def coefficient_lengths(self, n_samples: int) -> dict[str, int]:
        """Coefficient length per selected band, needed to size the network branches."""
        lengths = [
            len(c)
            for c in pywt.wavedec(
                np.zeros(n_samples), self.wavelet, level=self.level, mode=self.mode
            )
        ]
        return {band: lengths[band_index(band, self.level)] for band in self.bands}

    def to_dict(self) -> dict[str, Any]:
        return {
            "wavelet": self.wavelet,
            "level": self.level,
            "bands": list(self.bands),
            "mode": self.mode,
        }


@dataclass(frozen=True)
class WaveletDecomposition:
    """Selected wavelet bands of a batch of multi-channel windows."""

    bands: dict[str, np.ndarray]
    config: WaveletConfig

    def __getitem__(self, band: str) -> np.ndarray:
        try:
            return self.bands[band]
        except KeyError as exc:
            raise WaveletError(
                f"band {band!r} was not extracted; configured bands are {sorted(self.bands)}"
            ) from exc

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {band: array.shape for band, array in self.bands.items()}


def decompose(
    signal: np.ndarray,
    config: WaveletConfig,
    *,
    check_level: bool = True,
) -> WaveletDecomposition:
    """Decompose ``signal`` and return only the configured bands.

    Parameters
    ----------
    signal
        ``(n_samples,)``, ``(n_samples, n_channels)`` or ``(batch, n_channels, n_samples)``.
        The last layout is the one the models consume.
    config
        Wavelet settings.
    check_level
        Warn when the requested level exceeds what the window length supports, which
        produces coefficient arrays shorter than the wavelet filter.

    Returns
    -------
    WaveletDecomposition
        Each band has the same leading dimensions as ``signal`` with the time axis
        replaced by that band's coefficient length.

    Raises
    ------
    WaveletError
        On an unsupported shape or a failed transform. Failures are never silently
        replaced with zeros.
    """
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim == 1:
        batched = array[None, None, :]
        restore: str = "1d"
    elif array.ndim == 2:
        # (n_samples, n_channels) -> (1, n_channels, n_samples)
        batched = array.T[None, :, :]
        restore = "2d"
    elif array.ndim == 3:
        batched = array
        restore = "3d"
    else:
        raise WaveletError(f"unsupported signal shape {array.shape}")

    n_samples = batched.shape[-1]
    if check_level:
        usable = max_useful_level(n_samples, config.wavelet)
        if config.level > usable:
            logger.warning(
                "wavelet level %d exceeds the useful level %d for %d samples with %s; "
                "deep coefficients will be shorter than the filter",
                config.level,
                usable,
                n_samples,
                config.wavelet,
            )

    try:
        # pywt.wavedec vectorises over leading axes when given axis=-1.
        coeffs = pywt.wavedec(
            batched, config.wavelet, level=config.level, mode=config.mode, axis=-1
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed error, never swallowed
        raise WaveletError(
            f"wavelet decomposition failed for shape {array.shape} with "
            f"{config.wavelet!r} level {config.level}: {exc}"
        ) from exc

    selected: dict[str, np.ndarray] = {}
    for band in config.bands:
        values = coeffs[band_index(band, config.level)]
        if not np.isfinite(values).all():
            raise WaveletError(
                f"band {band!r} contains non-finite coefficients; the input window is "
                f"invalid and must be excluded, not zero-filled"
            )
        if restore == "1d":
            selected[band] = values[0, 0]
        elif restore == "2d":
            selected[band] = values[0].T
        else:
            selected[band] = values
    return WaveletDecomposition(bands=selected, config=config)
