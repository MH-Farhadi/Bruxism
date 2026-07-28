"""The single authoritative filtering implementation.

Training, evaluation, ablations, benchmarking and the diagnostic plots all call
:func:`apply_filter_chain` -- there is no second copy anywhere in the package.

Two properties matter scientifically:

**Filtering happens on the continuous recording, before windowing.** Filtering each
one-second window separately would inject a transient at both edges of *every* example.
:class:`bruxism.data.dataset.RecordingCache` enforces this ordering.

**Zero-phase filtering is offline.** The default ``zero_phase=True`` uses
:func:`scipy.signal.filtfilt`, which is acausal: it reads samples from the future of each
output sample. It cannot support any real-time, streaming or wearable claim. Set
``zero_phase=False`` for a causal ``lfilter`` chain if such a claim is ever needed; the
mode is recorded in every run bundle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, lfilter, sosfilt, sosfiltfilt, tf2sos

from bruxism.utils.logging import get_logger

__all__ = [
    "FilterChainConfig",
    "FilterDesignError",
    "FilterStage",
    "apply_filter_chain",
    "design_bandpass",
    "design_highpass",
    "design_notch",
    "validate_cutoffs",
]

logger = get_logger(__name__)


class FilterDesignError(ValueError):
    """Raised for a filter specification that cannot be realised at the given rate."""


def validate_cutoffs(
    cutoffs: float | tuple[float, float],
    sampling_rate: float,
    *,
    label: str = "filter",
) -> None:
    """Reject cutoffs that are non-positive, non-increasing, or at/above Nyquist.

    Designing a Butterworth filter with a normalised cutoff of exactly 1.0 produces an
    unstable or all-zero response depending on SciPy version, so the boundary is rejected
    rather than clipped.
    """
    nyquist = sampling_rate / 2.0
    values = (cutoffs,) if isinstance(cutoffs, int | float) else tuple(cutoffs)
    for value in values:
        if not np.isfinite(value) or value <= 0:
            raise FilterDesignError(f"{label}: cutoff {value} must be finite and strictly positive")
        if value >= nyquist:
            raise FilterDesignError(
                f"{label}: cutoff {value} Hz is at or above the Nyquist frequency "
                f"{nyquist} Hz (sampling rate {sampling_rate} Hz)"
            )
    if len(values) == 2 and values[0] >= values[1]:
        raise FilterDesignError(
            f"{label}: low cutoff {values[0]} Hz must be strictly below high cutoff {values[1]} Hz"
        )


def design_bandpass(
    low_hz: float, high_hz: float, sampling_rate: float, order: int = 4
) -> np.ndarray:
    """Second-order-section Butterworth bandpass. SOS is used for numerical stability."""
    validate_cutoffs((low_hz, high_hz), sampling_rate, label="bandpass")
    nyquist = sampling_rate / 2.0
    return butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band", output="sos")


def design_highpass(cutoff_hz: float, sampling_rate: float, order: int = 4) -> np.ndarray:
    """Second-order-section Butterworth high-pass."""
    validate_cutoffs(cutoff_hz, sampling_rate, label="highpass")
    return butter(order, cutoff_hz / (sampling_rate / 2.0), btype="high", output="sos")


def design_notch(freq_hz: float, sampling_rate: float, quality: float = 30.0) -> np.ndarray:
    """Second-order-section IIR notch at ``freq_hz``."""
    validate_cutoffs(freq_hz, sampling_rate, label="notch")
    if quality <= 0:
        raise FilterDesignError(f"notch: quality factor must be positive, got {quality}")
    b, a = iirnotch(freq_hz / (sampling_rate / 2.0), quality)
    return tf2sos(b, a)


@dataclass(frozen=True)
class FilterStage:
    """One stage of a filter chain.

    Attributes
    ----------
    kind
        ``"notch"``, ``"bandpass"`` or ``"highpass"``.
    rationale
        Why this stage exists. Required: a chain that stacks a 20-450 Hz bandpass with a
        separate 5 Hz high-pass (as the prototype did) is redundant, and the redundancy
        must be justified in writing rather than inherited silently.
    """

    kind: Literal["notch", "bandpass", "highpass"]
    rationale: str
    freq_hz: float | None = None
    low_hz: float | None = None
    high_hz: float | None = None
    order: int = 4
    quality: float = 30.0

    def design(self, sampling_rate: float) -> np.ndarray:
        if self.kind == "notch":
            if self.freq_hz is None:
                raise FilterDesignError("notch stage requires freq_hz")
            return design_notch(self.freq_hz, sampling_rate, self.quality)
        if self.kind == "bandpass":
            if self.low_hz is None or self.high_hz is None:
                raise FilterDesignError("bandpass stage requires low_hz and high_hz")
            return design_bandpass(self.low_hz, self.high_hz, sampling_rate, self.order)
        if self.kind == "highpass":
            if self.freq_hz is None:
                raise FilterDesignError("highpass stage requires freq_hz")
            return design_highpass(self.freq_hz, sampling_rate, self.order)
        raise FilterDesignError(f"unsupported filter kind {self.kind!r}")

    def describe(self) -> str:
        if self.kind == "notch":
            return f"notch {self.freq_hz:g} Hz (Q={self.quality:g})"
        if self.kind == "bandpass":
            return f"bandpass {self.low_hz:g}-{self.high_hz:g} Hz (order {self.order})"
        return f"highpass {self.freq_hz:g} Hz (order {self.order})"

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _default_emg_stages() -> list[FilterStage]:
    """The production EMG chain.

    Differs deliberately from the prototype, which applied a 60 Hz notch, a 20-450 Hz
    bandpass and *then* a 5 Hz high-pass. The 5 Hz high-pass is a no-op after a 20 Hz
    bandpass edge, so it is removed rather than carried forward. Whether the acquisition
    hardware already applied its own filtering is unknown (see ``docs/open_questions.md``
    Q9, the unexplained ``bandpass_filter: Index 143`` / ``notch_filter: Index 9``
    metadata fields), so the offline chain is kept minimal and explicit.
    """
    return [
        FilterStage(
            kind="notch",
            freq_hz=60.0,
            quality=30.0,
            rationale="Suppress North American mains interference at 60 Hz.",
        ),
        FilterStage(
            kind="bandpass",
            low_hz=20.0,
            high_hz=450.0,
            order=4,
            rationale=(
                "Standard surface-EMG band. The upper edge sits below the 600 Hz Nyquist "
                "frequency of the 1200 Hz acquisition rate."
            ),
        ),
    ]


def _default_mic_stages() -> list[FilterStage]:
    """The production microphone chain: DC removal only.

    The microphone channel is integer-valued with a large positive offset. A high-pass at
    20 Hz removes that offset without touching the chewing/grinding band, and no further
    shaping is applied because the transducer's response is undocumented.
    """
    return [
        FilterStage(
            kind="highpass",
            freq_hz=20.0,
            order=2,
            rationale=(
                "Remove the large DC offset of the microphone channel. No further shaping "
                "is applied because the transducer response and units are undocumented."
            ),
        )
    ]


@dataclass(frozen=True)
class FilterChainConfig:
    """Complete, reviewable specification of the offline filtering applied to a recording.

    Attributes
    ----------
    zero_phase
        ``True`` uses :func:`scipy.signal.sosfiltfilt` (acausal, offline). ``False`` uses
        :func:`scipy.signal.sosfilt` (causal, streaming-compatible but phase-distorting).
    padlen
        Edge padding for the zero-phase path. ``None`` lets SciPy choose. Applied to the
        continuous recording, so it affects only the recording's own boundaries.
    """

    emg_stages: tuple[FilterStage, ...] = field(
        default_factory=lambda: tuple(_default_emg_stages())
    )
    mic_stages: tuple[FilterStage, ...] = field(
        default_factory=lambda: tuple(_default_mic_stages())
    )
    zero_phase: bool = True
    padlen: int | None = None

    @property
    def is_causal(self) -> bool:
        """Whether this chain could run on a live stream."""
        return not self.zero_phase

    @property
    def realtime_claim_supported(self) -> bool:
        """Alias with the name a reviewer would look for. Always mirrors :attr:`is_causal`."""
        return self.is_causal

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "zero_phase_offline (acausal)" if self.zero_phase else "causal_streaming",
            "supports_realtime_claim": self.realtime_claim_supported,
            "emg": [stage.describe() for stage in self.emg_stages],
            "mic": [stage.describe() for stage in self.mic_stages],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "emg_stages": [stage.to_dict() for stage in self.emg_stages],
            "mic_stages": [stage.to_dict() for stage in self.mic_stages],
            "zero_phase": self.zero_phase,
            "padlen": self.padlen,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FilterChainConfig:
        def stages(items: list[dict[str, Any]] | None) -> tuple[FilterStage, ...]:
            return tuple(FilterStage(**item) for item in (items or []))

        return cls(
            emg_stages=stages(payload.get("emg_stages")) or tuple(_default_emg_stages()),
            mic_stages=stages(payload.get("mic_stages")) or tuple(_default_mic_stages()),
            zero_phase=bool(payload.get("zero_phase", True)),
            padlen=payload.get("padlen"),
        )


def _apply_stages(
    signal: np.ndarray,
    stages: tuple[FilterStage, ...],
    sampling_rate: float,
    *,
    zero_phase: bool,
    padlen: int | None,
) -> np.ndarray:
    out = np.asarray(signal, dtype=np.float64)
    if out.ndim == 1:
        out = out[:, None]
        squeeze = True
    elif out.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"expected a 1-D or 2-D (time, channels) array, got shape {out.shape}")

    for stage in stages:
        sos = stage.design(sampling_rate)
        min_len = 3 * (sos.shape[0] * 2)
        if zero_phase and out.shape[0] <= min_len:
            raise FilterDesignError(
                f"{stage.describe()}: signal of {out.shape[0]} samples is too short for "
                f"zero-phase filtering (needs more than {min_len}); filter the continuous "
                f"recording rather than individual windows"
            )
        if zero_phase:
            out = sosfiltfilt(sos, out, axis=0, padlen=padlen)
        else:
            out = sosfilt(sos, out, axis=0)
    return out[:, 0] if squeeze else out


def apply_filter_chain(
    signal: np.ndarray,
    config: FilterChainConfig,
    sampling_rate: float,
    *,
    modality: Literal["emg", "mic"],
) -> np.ndarray:
    """Filter one continuous recording's signal.

    Parameters
    ----------
    signal
        ``(n_samples,)`` or ``(n_samples, n_channels)`` array of raw samples. Time is
        always the **first** axis.
    config
        The chain to apply.
    sampling_rate
        Hz. Every cutoff is validated against this rate's Nyquist frequency.
    modality
        Selects ``config.emg_stages`` or ``config.mic_stages``.

    Returns
    -------
    numpy.ndarray
        Filtered ``float64`` array with the same shape as ``signal``.

    Raises
    ------
    FilterDesignError
        On an invalid cutoff, or a signal too short for the requested zero-phase chain.
    """
    stages = config.emg_stages if modality == "emg" else config.mic_stages
    return _apply_stages(
        signal,
        stages,
        sampling_rate,
        zero_phase=config.zero_phase,
        padlen=config.padlen,
    )


# Kept for the legacy-parity harness only: the prototype's exact transfer-function path
# (``butter(..., output='ba')`` + ``filtfilt``), which differs from the SOS path by
# floating-point round-off. Never call this from production code.
def _legacy_ba_filtfilt(
    signal: np.ndarray, b: np.ndarray, a: np.ndarray, *, zero_phase: bool = True
) -> np.ndarray:
    """Reproduce the prototype's transfer-function filtering, for parity checks only."""
    return filtfilt(b, a, signal, axis=0) if zero_phase else lfilter(b, a, signal, axis=0)
