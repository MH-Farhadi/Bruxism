"""Explicitly defined time and time-frequency features for the classical baselines.

Every feature has a name, a formula and a test. Two corrections relative to the research
prototype:

* **Median frequency is the true spectral median.** The prototype computed a statistic
  that was not the frequency below which half the power spectrum lies. Here
  :func:`median_frequency` finds the frequency at which the cumulative power spectral
  density reaches half of its total, verified in the unit tests against a synthetic
  spectrum with a known median.
* **Failures raise.** The prototype wrapped its wavelet-packet and continuous-wavelet
  blocks in bare ``except`` handlers that appended four zeros per channel, so a broken
  transform was indistinguishable from a genuinely flat signal. Nothing here substitutes
  zeros; a failed extraction raises :class:`~bruxism.preprocessing.wavelets.WaveletError`
  or :class:`ValueError` and the sample is excluded.

The prototype's continuous-wavelet block called ``pywt.cwt(sig, scales, 'db4', ...)``.
``db4`` is a discrete wavelet and is not a valid continuous-wavelet argument, so that
block could only ever have produced the zero fallback. It is removed rather than repaired:
nothing in the manuscript depends on it. Wavelet-packet features are implemented properly
at the configured depth instead of the prototype's first-level-only ``['a', 'd']`` pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pywt
from scipy.signal import welch

from bruxism.preprocessing.wavelets import WaveletConfig, band_frequencies, decompose
from bruxism.utils.logging import get_logger

__all__ = [
    "FeatureConfig",
    "FeatureExtractor",
    "band_power_ratios",
    "median_frequency",
    "spectral_edge_frequency",
    "wavelet_packet_energies",
    "zero_crossing_rate",
]

logger = get_logger(__name__)


def median_frequency(
    signal: np.ndarray, sampling_rate: float, *, nperseg: int | None = None
) -> float:
    """Frequency below which half the signal's power lies.

    Computed from a Welch power spectral density estimate: the returned value is the
    frequency ``f_med`` at which the cumulative PSD first reaches half the total power,
    linearly interpolated between the bracketing bins.

    Raises
    ------
    ValueError
        If the signal is empty or carries no power.
    """
    values = np.asarray(signal, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("median_frequency requires a non-empty signal")
    segment = nperseg if nperseg is not None else min(256, values.size)
    freqs, psd = welch(values, fs=sampling_rate, nperseg=segment)
    total = float(np.trapezoid(psd, freqs))
    if total <= 0:
        raise ValueError(
            "signal has zero total power; median frequency is undefined and must not be "
            "reported as 0 Hz"
        )
    cumulative = np.concatenate([[0.0], np.cumsum(np.diff(freqs) * (psd[1:] + psd[:-1]) / 2)])
    half = total / 2.0
    upper = int(np.searchsorted(cumulative, half))
    if upper == 0:
        return float(freqs[0])
    if upper >= freqs.size:
        return float(freqs[-1])
    lower = upper - 1
    span = cumulative[upper] - cumulative[lower]
    if span <= 0:
        return float(freqs[upper])
    weight = (half - cumulative[lower]) / span
    return float(freqs[lower] + weight * (freqs[upper] - freqs[lower]))


def spectral_edge_frequency(
    signal: np.ndarray, sampling_rate: float, *, fraction: float = 0.95
) -> float:
    """Frequency below which ``fraction`` of the total power lies."""
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    values = np.asarray(signal, dtype=np.float64).ravel()
    freqs, psd = welch(values, fs=sampling_rate, nperseg=min(256, values.size))
    cumulative = np.cumsum(psd)
    total = cumulative[-1]
    if total <= 0:
        raise ValueError("signal has zero total power; spectral edge frequency is undefined")
    return float(freqs[int(np.searchsorted(cumulative, fraction * total))])


def zero_crossing_rate(signal: np.ndarray, *, threshold: float = 0.0) -> float:
    """Fraction of adjacent sample pairs that cross ``threshold``."""
    values = np.asarray(signal, dtype=np.float64).ravel() - threshold
    if values.size < 2:
        return 0.0
    return float(np.mean(np.diff(np.signbit(values)) != 0))


def band_power_ratios(
    signal: np.ndarray, config: WaveletConfig, sampling_rate: float
) -> dict[str, float]:
    """Relative energy of each configured wavelet band, keyed by band name and Hz range."""
    decomposition = decompose(signal, config, check_level=False)
    energies = {band: float(np.sum(decomposition[band] ** 2)) for band in config.bands}
    total = sum(energies.values())
    if total <= 0:
        raise ValueError("all wavelet bands carry zero energy; the window is degenerate")
    out: dict[str, float] = {}
    for band, energy in energies.items():
        low, high = band_frequencies(band, config.level, sampling_rate)
        out[f"relenergy_{band}_{low:.0f}-{high:.0f}Hz"] = energy / total
    return out


def wavelet_packet_energies(
    signal: np.ndarray, *, wavelet: str = "db4", max_level: int = 3
) -> dict[str, float]:
    """Relative energy of every wavelet-packet node at ``max_level``.

    Unlike the prototype -- which declared ``maxlevel=4`` but only read the two
    first-level nodes ``'a'`` and ``'d'`` -- this walks the full natural-order node set at
    the requested depth, producing ``2**max_level`` features.
    """
    values = np.asarray(signal, dtype=np.float64).ravel()
    packet = pywt.WaveletPacket(values, wavelet, maxlevel=max_level)
    nodes = packet.get_level(max_level, order="natural")
    energies = {node.path: float(np.sum(np.asarray(node.data) ** 2)) for node in nodes}
    total = sum(energies.values())
    if total <= 0:
        raise ValueError("all wavelet-packet nodes carry zero energy; the window is degenerate")
    return {f"wpt_{path}": energy / total for path, energy in energies.items()}


@dataclass(frozen=True)
class FeatureConfig:
    """Which feature groups the classical baselines receive."""

    sampling_rate: float = 1200.0
    wavelet: WaveletConfig = WaveletConfig(
        wavelet="db4", level=4, bands=("A4", "D4", "D3", "D2", "D1")
    )
    include_time_domain: bool = True
    include_spectral: bool = True
    include_wavelet_bands: bool = True
    include_wavelet_packets: bool = True
    packet_level: int = 3
    include_cross_channel: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampling_rate": self.sampling_rate,
            "wavelet": self.wavelet.to_dict(),
            "include_time_domain": self.include_time_domain,
            "include_spectral": self.include_spectral,
            "include_wavelet_bands": self.include_wavelet_bands,
            "include_wavelet_packets": self.include_wavelet_packets,
            "packet_level": self.packet_level,
            "include_cross_channel": self.include_cross_channel,
        }


class FeatureExtractor:
    """Turns one EMG/microphone window pair into a named, ordered feature vector.

    The feature order is fixed by :attr:`feature_names` and is identical for every window,
    so a fitted scikit-learn model can be applied to a new run without re-deriving it.
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self._names: list[str] | None = None

    def _channel_features(self, signal: np.ndarray, prefix: str) -> dict[str, float]:
        cfg = self.config
        out: dict[str, float] = {}
        values = np.asarray(signal, dtype=np.float64).ravel()

        if cfg.include_time_domain:
            out[f"{prefix}_rms"] = float(np.sqrt(np.mean(values**2)))
            out[f"{prefix}_mav"] = float(np.mean(np.abs(values)))
            out[f"{prefix}_std"] = float(np.std(values))
            out[f"{prefix}_waveform_length"] = float(np.sum(np.abs(np.diff(values))))
            out[f"{prefix}_zero_crossing_rate"] = zero_crossing_rate(values)
            out[f"{prefix}_kurtosis"] = _kurtosis(values)
            out[f"{prefix}_skewness"] = _skewness(values)

        if cfg.include_spectral:
            out[f"{prefix}_median_frequency_hz"] = median_frequency(values, cfg.sampling_rate)
            out[f"{prefix}_spectral_edge_95_hz"] = spectral_edge_frequency(
                values, cfg.sampling_rate, fraction=0.95
            )

        if cfg.include_wavelet_bands:
            for name, value in band_power_ratios(values, cfg.wavelet, cfg.sampling_rate).items():
                out[f"{prefix}_{name}"] = value

        if cfg.include_wavelet_packets:
            for name, value in wavelet_packet_energies(
                values, wavelet=cfg.wavelet.wavelet, max_level=cfg.packet_level
            ).items():
                out[f"{prefix}_{name}"] = value
        return out

    def extract_named(self, emg: np.ndarray, mic: np.ndarray) -> dict[str, float]:
        """Extract all features as an ordered ``name -> value`` mapping.

        Parameters
        ----------
        emg
            ``(n_samples, n_channels)`` window.
        mic
            ``(n_samples,)`` window.
        """
        emg = np.asarray(emg, dtype=np.float64)
        mic = np.asarray(mic, dtype=np.float64).ravel()
        if emg.ndim != 2:
            raise ValueError(f"emg must be (n_samples, n_channels), got shape {emg.shape}")

        features: dict[str, float] = {}
        for channel in range(emg.shape[1]):
            features.update(self._channel_features(emg[:, channel], f"emg{channel + 1}"))
        features.update(self._channel_features(mic, "mic"))

        if self.config.include_cross_channel and emg.shape[1] > 1:
            for i in range(emg.shape[1]):
                for j in range(i + 1, emg.shape[1]):
                    features[f"corr_emg{i + 1}_emg{j + 1}"] = _safe_corr(emg[:, i], emg[:, j])
            for i in range(emg.shape[1]):
                features[f"corr_emg{i + 1}_mic"] = _safe_corr(emg[:, i], mic)

        values = np.fromiter(features.values(), dtype=np.float64, count=len(features))
        if not np.isfinite(values).all():
            bad = [name for name, value in features.items() if not np.isfinite(value)]
            raise ValueError(
                f"non-finite feature values for {bad}; the window is invalid and must be "
                f"excluded rather than zero-filled"
            )
        if self._names is None:
            self._names = list(features)
        elif list(features) != self._names:
            raise ValueError("feature name order changed between windows; this is a bug")
        return features

    def extract(self, emg: np.ndarray, mic: np.ndarray) -> np.ndarray:
        """Feature vector in :attr:`feature_names` order."""
        return np.fromiter(self.extract_named(emg, mic).values(), dtype=np.float64)

    @property
    def feature_names(self) -> list[str]:
        """Names in extraction order. Available after the first :meth:`extract` call."""
        if self._names is None:
            raise RuntimeError("call extract() at least once before reading feature_names")
        return list(self._names)

    def n_features(self, n_emg_channels: int = 4) -> int:
        """Feature-vector length, computed by extracting from a synthetic probe window."""
        rng = np.random.default_rng(0)
        n = 1024
        probe_emg = rng.standard_normal((n, n_emg_channels))
        probe_mic = rng.standard_normal(n)
        return len(self.extract(probe_emg, probe_mic))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, returning 0.0 only when a channel is genuinely constant."""
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _kurtosis(values: np.ndarray) -> float:
    centred = values - values.mean()
    variance = float(np.mean(centred**2))
    if variance <= 0:
        return 0.0
    return float(np.mean(centred**4) / variance**2)


def _skewness(values: np.ndarray) -> float:
    centred = values - values.mean()
    variance = float(np.mean(centred**2))
    if variance <= 0:
        return 0.0
    return float(np.mean(centred**3) / variance**1.5)
