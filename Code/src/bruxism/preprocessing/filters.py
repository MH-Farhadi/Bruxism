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

**The EMG chain notches every mains harmonic, not just the fundamental.** Until
2026-08-03 it notched 60 Hz and band-passed 20-450 Hz. The acquisition hardware had
*already* removed 60 Hz (``notch_filter: Index 9`` in every metadata sidecar), so the one
stage that did anything removed a frequency that was not there, while 180, 300 and 420 Hz
-- measured at 37,000x to 846,000x the local noise floor -- passed through untouched and
made up 85-99 % of the in-band power of the "filtered" signal. See ``cause.md`` and
``docs/open_questions.md`` Q9. Anyone reusing the textbook "notch the mains fundamental,
then band-pass" recipe on hardware that already notches has the same defect.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Final, Literal

import numpy as np
from scipy.signal import (
    butter,
    filtfilt,
    iircomb,
    iirnotch,
    lfilter,
    sosfilt,
    sosfiltfilt,
    tf2sos,
)

from bruxism.utils.logging import get_logger

__all__ = [
    "DEFAULT_EMG_BAND_HZ",
    "DEFAULT_MAINS_HZ",
    "DEFAULT_NOTCH_BANDWIDTH_HZ",
    "DEFAULT_NOTCH_QUALITY",
    "FilterChainConfig",
    "FilterDesignError",
    "FilterStage",
    "apply_filter_chain",
    "chain_magnitude",
    "design_bandpass",
    "design_comb",
    "design_highpass",
    "design_notch",
    "emg_stages",
    "mains_harmonics",
    "mic_envelope_stages",
    "validate_cutoffs",
    "white_noise_power_gain",
]

logger = get_logger(__name__)

#: Mains fundamental of the acquisition site (North American supply).
DEFAULT_MAINS_HZ: Final[float] = 60.0

#: Quality factor used only when a constant-Q chain is requested explicitly.
#: ``iirnotch`` gives a -3 dB width of ``f0 / Q``, so Q=30 costs 2 Hz at 60 Hz and 14 Hz at
#: 420 Hz.
DEFAULT_NOTCH_QUALITY: Final[float] = 30.0

#: Absolute -3 dB width of every mains notch, in Hz. Each notch's Q is derived from it, so
#: all seven are equally wide.
#:
#: Constant Q is the textbook choice and it is the wrong one here. It gives 2 Hz at 60 Hz
#: and 14 Hz at 420 Hz, while the interference is spread by roughly the same absolute
#: amount at every harmonic -- so constant Q is too narrow exactly where the residue is and
#: too wide where there is nothing left to remove. Measured on the four worst-contaminated
#: recordings, at an identical 13 % total band cost, 8 Hz constant-width notches leave
#: S02's worst cell at 5.2 % harmonic power and 1.8x its local floor, against 35.0 % and
#: 21.8x for Q=30. The 60 Hz notch matters despite the hardware notch: the hardware left
#: a residue 40x above the local floor, spread wider than the 2 Hz a Q=30 notch covers.
DEFAULT_NOTCH_BANDWIDTH_HZ: Final[float] = 8.0

#: Surface-EMG passband. The upper edge sits below the 600 Hz Nyquist of the 1200 Hz rate.
DEFAULT_EMG_BAND_HZ: Final[tuple[float, float]] = (20.0, 450.0)


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


def design_comb(freq_hz: float, sampling_rate: float, quality: float = 30.0) -> np.ndarray:
    """Second-order-section IIR notching comb with nulls at every multiple of ``freq_hz``.

    One stage replaces a bank of individual notches and is cheaper to run, but it notches
    *every* multiple up to Nyquist -- including any that lie outside the band of interest
    -- and :func:`scipy.signal.iircomb` requires ``sampling_rate / freq_hz`` to be an
    integer, so it cannot express an arbitrary mains frequency at an arbitrary rate.
    """
    validate_cutoffs(freq_hz, sampling_rate, label="comb")
    if quality <= 0:
        raise FilterDesignError(f"comb: quality factor must be positive, got {quality}")
    ratio = sampling_rate / freq_hz
    if abs(ratio - round(ratio)) > 1e-9:
        raise FilterDesignError(
            f"comb: sampling rate {sampling_rate} Hz is not an integer multiple of "
            f"{freq_hz} Hz (ratio {ratio:.4f}); scipy.signal.iircomb cannot realise this "
            f"comb. Use a notch bank instead."
        )
    b, a = iircomb(freq_hz, quality, ftype="notch", fs=sampling_rate)
    return tf2sos(b, a)


def mains_harmonics(
    mains_hz: float,
    sampling_rate: float,
    *,
    low_hz: float = 0.0,
    high_hz: float | None = None,
) -> tuple[float, ...]:
    """Every multiple of ``mains_hz`` inside ``(low_hz, high_hz]`` and below Nyquist.

    The fundamental is included when it falls in the interval. It is kept even though the
    acquisition hardware already removed it here: the chain must not depend on a property
    of one particular amplifier, and notching an absent frequency costs 2 Hz of a 430 Hz
    band.
    """
    if mains_hz <= 0:
        raise FilterDesignError(f"mains frequency must be positive, got {mains_hz}")
    ceiling = sampling_rate / 2.0 if high_hz is None else min(high_hz, sampling_rate / 2.0)
    harmonics: list[float] = []
    index = 1
    while mains_hz * index < ceiling:
        frequency = mains_hz * index
        if frequency > low_hz:
            harmonics.append(frequency)
        index += 1
    return tuple(harmonics)


StageKind = Literal["notch", "bandpass", "highpass", "comb", "spectral_interpolation"]

#: Stages that are realised as second-order sections and run through ``sosfilt(filt)``.
_SOS_KINDS: Final[frozenset[str]] = frozenset({"notch", "bandpass", "highpass", "comb"})


@dataclass(frozen=True)
class FilterStage:
    """One stage of a filter chain.

    Attributes
    ----------
    kind
        ``"notch"``, ``"bandpass"``, ``"highpass"``, ``"comb"`` (nulls at every multiple of
        ``freq_hz``) or ``"spectral_interpolation"`` (magnitude of the bins around each
        harmonic of ``freq_hz`` replaced by the surrounding noise floor, original phase
        retained).
    rationale
        Why this stage exists. Required: a chain that stacks a 20-450 Hz bandpass with a
        separate 5 Hz high-pass (as the prototype did) is redundant, and the redundancy
        must be justified in writing rather than inherited silently.
    half_width_hz
        Half-width of the replaced band, ``spectral_interpolation`` only.
    """

    kind: StageKind
    rationale: str
    freq_hz: float | None = None
    low_hz: float | None = None
    high_hz: float | None = None
    order: int = 4
    quality: float = 30.0
    half_width_hz: float = 1.5

    @property
    def is_sos(self) -> bool:
        """Whether this stage is realised as second-order sections."""
        return self.kind in _SOS_KINDS

    @property
    def is_acausal(self) -> bool:
        """Whether this stage needs the whole recording regardless of ``zero_phase``."""
        return self.kind == "spectral_interpolation"

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
        if self.kind == "comb":
            if self.freq_hz is None:
                raise FilterDesignError("comb stage requires freq_hz")
            return design_comb(self.freq_hz, sampling_rate, self.quality)
        if self.kind == "spectral_interpolation":
            raise FilterDesignError(
                "spectral_interpolation is not a linear time-invariant filter and has no "
                "second-order-section form; call magnitude_response() for a plottable "
                "response, or apply_filter_chain() to apply it"
            )
        raise FilterDesignError(f"unsupported filter kind {self.kind!r}")

    def harmonics(self, sampling_rate: float) -> tuple[float, ...]:
        """Frequencies this stage suppresses, for ``comb`` and ``spectral_interpolation``."""
        if self.freq_hz is None:
            raise FilterDesignError(f"{self.kind} stage requires freq_hz")
        return mains_harmonics(
            self.freq_hz,
            sampling_rate,
            low_hz=self.low_hz if self.low_hz is not None else 0.0,
            high_hz=self.high_hz,
        )

    def magnitude_response(self, frequencies: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Single-pass magnitude response at ``frequencies``, for the filter figure.

        ``spectral_interpolation`` is not an LTI filter; the curve returned for it is the
        *intended* rectangular suppression, drawn so a reader can see which bins the stage
        replaces. The figure's caption says so.
        """
        grid = np.asarray(frequencies, dtype=np.float64)
        if self.is_sos:
            from scipy.signal import sosfreqz

            _, response = sosfreqz(self.design(sampling_rate), worN=grid, fs=sampling_rate)
            return np.abs(response)
        magnitude = np.ones_like(grid)
        for harmonic in self.harmonics(sampling_rate):
            magnitude[np.abs(grid - harmonic) <= self.half_width_hz] = 0.0
        return magnitude

    def describe(self) -> str:
        if self.kind == "notch":
            return f"notch {self.freq_hz:g} Hz (Q={self.quality:g})"
        if self.kind == "bandpass":
            return f"bandpass {self.low_hz:g}-{self.high_hz:g} Hz (order {self.order})"
        if self.kind == "comb":
            return f"comb, nulls at n x {self.freq_hz:g} Hz (Q={self.quality:g})"
        if self.kind == "spectral_interpolation":
            return (
                f"spectral interpolation at n x {self.freq_hz:g} Hz (+/-{self.half_width_hz:g} Hz)"
            )
        return f"highpass {self.freq_hz:g} Hz (order {self.order})"

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


#: Rationale shared by every mains notch, so the reason survives into every run bundle.
_MAINS_RATIONALE: Final[str] = (
    "Suppress mains interference. The acquisition hardware had already removed the "
    "fundamental (notch_filter: Index 9 in every metadata sidecar), so the surviving "
    "interference was entirely at the harmonics: 180, 300 and 420 Hz measured 37,000x to "
    "846,000x above the local noise floor and made up 85-99 % of the in-band power of the "
    "previously 'filtered' signal (cause.md, open_questions.md Q9). A chain that notches "
    "only the fundamental removes the one frequency that is not there."
)


def emg_stages(
    *,
    mains_hz: float = DEFAULT_MAINS_HZ,
    notch_harmonics: bool = True,
    quality: float = DEFAULT_NOTCH_QUALITY,
    notch_bandwidth_hz: float | None = DEFAULT_NOTCH_BANDWIDTH_HZ,
    band_hz: tuple[float, float] = DEFAULT_EMG_BAND_HZ,
    sampling_rate: float = 1200.0,
) -> list[FilterStage]:
    """Build the production EMG chain: mains notches, then the surface-EMG bandpass.

    Parameters
    ----------
    mains_hz
        Supply frequency. 60 Hz in North America, 50 Hz elsewhere.
    notch_harmonics
        ``True`` notches every multiple of ``mains_hz`` inside the passband; ``False``
        notches the fundamental only, which is what produced the defect in ``cause.md``
        and is kept solely so that regression comparisons against the old chain can be
        expressed in configuration rather than in patched code.
    notch_bandwidth_hz
        Absolute -3 dB width of every notch. Each notch's Q is derived as
        ``freq / notch_bandwidth_hz``, so all seven are equally wide. This is the default
        because the interference spread here is absolute rather than proportional: see
        :data:`DEFAULT_NOTCH_BANDWIDTH_HZ`. Set to ``None`` to use a constant ``quality``
        instead.
    quality
        Constant notch quality factor, used only when ``notch_bandwidth_hz`` is ``None``.
        The -3 dB width is then ``f0 / quality`` -- 2 Hz at 60 Hz and 14 Hz at 420 Hz.
    band_hz
        Bandpass edges. Notches are placed only inside this band.
    sampling_rate
        Rate the chain will run at, used to enumerate the harmonics below Nyquist.
    """
    low_hz, high_hz = band_hz
    frequencies = (
        mains_harmonics(mains_hz, sampling_rate, low_hz=low_hz, high_hz=high_hz)
        if notch_harmonics
        else (mains_hz,)
    )
    stages = [
        FilterStage(
            kind="notch",
            freq_hz=frequency,
            quality=(frequency / notch_bandwidth_hz if notch_bandwidth_hz is not None else quality),
            rationale=_MAINS_RATIONALE,
        )
        for frequency in frequencies
    ]
    stages.append(
        FilterStage(
            kind="bandpass",
            low_hz=low_hz,
            high_hz=high_hz,
            order=4,
            rationale=(
                "Standard surface-EMG band. The upper edge sits below the 600 Hz Nyquist "
                "frequency of the 1200 Hz acquisition rate."
            ),
        )
    )
    return stages


def _default_emg_stages() -> list[FilterStage]:
    """The production EMG chain at the nominal 1200 Hz rate.

    Notches every multiple of the 60 Hz mains inside the 20-450 Hz passband -- 60, 120,
    180, 240, 300, 360 and 420 Hz -- then band-passes. Until 2026-08-03 this notched 60 Hz
    only; see :data:`_MAINS_RATIONALE` and ``cause.md`` for why that removed the one
    frequency the hardware had already taken out and left the three that dominated the
    recordings.

    Also differs from the prototype, which applied a 60 Hz notch, a 20-450 Hz bandpass and
    *then* a 5 Hz high-pass. The 5 Hz high-pass is a no-op after a 20 Hz bandpass edge, so
    it is removed rather than carried forward.
    """
    return emg_stages()


#: Rationale carried by the published microphone chain, including what it was later
#: measured to cost. Kept as the default so that the runs already reported reproduce
#: byte-for-byte; the measurement is attached so no reader can mistake it for a considered
#: design choice.
_MIC_HIGHPASS_RATIONALE: Final[str] = (
    "Remove the large DC offset of the microphone channel. No further shaping is applied "
    "because the transducer response and units are undocumented. MEASURED COST, added "
    "2026-08-12 after the audit in audio.md: 96 % of this channel's raw power lies below "
    "10 Hz (range 91.4-98.7 % over the 100 recordings) and all of its condition separation "
    "is at 1-3 Hz -- the chew-rhythm band. This 20 Hz high-pass therefore retains a median "
    "of 1.19 % of the channel's variance (range 0.50-3.25 %) and discards precisely the "
    "band that carries the information. What survives is at or below the analog-to-digital "
    "quantisation floor for the clenching and grinding conditions. The stage is RETAINED AS "
    "THE DEFAULT only because the reported runs used it; it is not recommended. Use "
    "mic_envelope_stages() for any new analysis of this transducer."
)


def _default_mic_stages() -> list[FilterStage]:
    """The published microphone chain: a 20 Hz high-pass, and nothing else.

    This is the chain every reported run used. It is kept as the default so those runs
    reproduce exactly, and it is documented as defective rather than quietly replaced --
    see :data:`_MIC_HIGHPASS_RATIONALE` and ``audio.md`` 1.3-1.4 for what it costs.

    Note also that the mic branch of the model reads a wavelet band, ``A5``, whose nominal
    range is 0-18.75 Hz at 1200 Hz -- entirely inside this stage's stopband. That branch is
    therefore fed the high-pass roll-off residue. The contradiction is now caught by
    :func:`assert_bands_within_passband`; the published configurations declare it
    explicitly instead of silently reintroducing it.
    """
    return [FilterStage(kind="highpass", freq_hz=20.0, order=2, rationale=_MIC_HIGHPASS_RATIONALE)]


def mic_envelope_stages(
    *,
    drift_hz: float = 0.2,
    ceiling_hz: float = 20.0,
    sampling_rate: float = 1200.0,
) -> list[FilterStage]:
    """The chain this transducer actually needs: keep the modulation band, drop the drift.

    The channel measured in ``audio.md`` is a rectified, smoothed sound-level output, not
    an acoustic waveform: 96 % of its power is below 10 Hz and the separation between
    conditions lives at 1-3 Hz (chewing 63.2, movement 41.7, rest 16.6, clenching 4.2,
    grinding 3.8, in raw counts squared). The published chain
    (:func:`_default_mic_stages`) removes that band and keeps the quantisation noise above
    it. This one does the opposite.

    Parameters
    ----------
    drift_hz
        High-pass edge. Low enough to pass the slowest chew rhythm (~0.5 Hz) and high
        enough to remove baseline wander and the DC offset.
    ceiling_hz
        Low-pass edge. Above the fastest masticatory rhythm and below the band in which
        this channel carries only dither.

    Notes
    -----
    **This does not rescue the 2025-08 collection.** The duplication and misalignment
    documented in ``audio.md`` 1.1-1.2 are independent of any filter, and no chain can
    undo them. This builder exists for a future collection with the same transducer, and
    for measuring how much of the discarded band was real.
    """
    validate_cutoffs((drift_hz, ceiling_hz), sampling_rate, label="mic envelope chain")
    return [
        FilterStage(
            kind="highpass",
            freq_hz=drift_hz,
            order=2,
            rationale=(
                "Remove the DC offset and baseline wander of the sound-level channel while "
                "passing the masticatory modulation band. Placed at 0.2 Hz rather than "
                "20 Hz because the discriminative content of this transducer is at 1-3 Hz "
                "(audio.md 1.3), which a 20 Hz edge deletes."
            ),
        ),
        FilterStage(
            kind="bandpass",
            low_hz=drift_hz,
            high_hz=ceiling_hz,
            order=4,
            rationale=(
                "Restrict to the modulation band of a rectified sound-level output. Above "
                f"{ceiling_hz:g} Hz this channel carries quantisation noise rather than "
                "signal: with a 1.0-count step, 42 of the 100 recordings audited are within "
                "3 dB of the quantisation floor in the band the published chain kept."
            ),
        ),
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
        """Whether this chain could run on a live stream.

        ``zero_phase=False`` is necessary but not sufficient: a spectral-interpolation
        stage reads the whole recording to estimate the interference, so a chain
        containing one is acausal however the IIR stages are applied.
        """
        acausal_stage = any(stage.is_acausal for stage in (*self.emg_stages, *self.mic_stages))
        return not self.zero_phase and not acausal_stage

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
        """Build a chain from a mapping, accepting either form of the EMG declaration.

        ``emg_stages`` lists every stage explicitly. ``emg_mains`` declares the *intent*
        -- supply frequency, whether to notch the harmonics, notch Q, passband -- and
        expands to the same stages. The compact form exists because the corrected chain
        needs seven notches and hand-copying seven near-identical YAML blocks into five
        config files is precisely where a wrong frequency would enter unnoticed.
        :meth:`to_dict` always writes the expanded list, so the configuration hash and the
        run bundle are identical whichever form was used.
        """

        def stages(items: list[dict[str, Any]] | None) -> tuple[FilterStage, ...]:
            return tuple(FilterStage(**item) for item in (items or []))

        emg = stages(payload.get("emg_stages"))
        mains = payload.get("emg_mains")
        if mains is not None:
            if emg:
                raise FilterDesignError(
                    "specify either 'emg_stages' or 'emg_mains', not both; the compact "
                    "form expands to a full stage list and the two would silently disagree"
                )
            known = {
                "mains_hz",
                "notch_harmonics",
                "quality",
                "notch_bandwidth_hz",
                "band_hz",
                "sampling_rate",
            }
            unknown = sorted(set(mains) - known)
            if unknown:
                raise FilterDesignError(
                    f"unknown key(s) {unknown} under 'filters.emg_mains'; "
                    f"valid keys are {sorted(known)}"
                )
            options = dict(mains)
            if "band_hz" in options:
                options["band_hz"] = tuple(options["band_hz"])
            emg = tuple(emg_stages(**options))

        return cls(
            emg_stages=emg or tuple(_default_emg_stages()),
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
        if stage.kind == "spectral_interpolation":
            out = _apply_spectral_interpolation(out, stage, sampling_rate)
            continue
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


def _apply_spectral_interpolation(
    signal: np.ndarray, stage: FilterStage, sampling_rate: float
) -> np.ndarray:
    """Replace the magnitude of each harmonic's bins with the surrounding noise floor.

    The phase of every bin is left untouched, so only the *excess* magnitude at the
    interference frequencies is removed and the broadband EMG that shares those bins keeps
    its timing. This preserves bandwidth, which a notch bank does not: a Q=30 notch at
    420 Hz deletes 14 Hz of signal, this deletes the interference and leaves an estimate of
    the EMG that was underneath it.

    Deterministic: the replacement magnitude is the median of the two reference bands
    flanking each harmonic, never a random draw.
    """
    n_samples = signal.shape[0]
    half_width = float(stage.half_width_hz)
    resolution = sampling_rate / n_samples
    if half_width < 2 * resolution:
        raise FilterDesignError(
            f"{stage.describe()}: a signal of {n_samples} samples resolves "
            f"{resolution:.3f} Hz, too coarse for a +/-{half_width:g} Hz band; apply "
            f"spectral interpolation to the continuous recording, not to a window"
        )

    spectrum = np.fft.rfft(signal, axis=0)
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)
    magnitude = np.abs(spectrum)
    # A zero-magnitude bin has no defined phase; 1.0 keeps the reconstruction real and the
    # bin's (zero) contribution unchanged.
    phase = np.divide(spectrum, magnitude, out=np.ones_like(spectrum), where=magnitude > 0)

    # Reference floor is read from the bins just outside the replaced band on both sides.
    reference_width = 5.0 * half_width
    for harmonic in stage.harmonics(sampling_rate):
        offset = np.abs(frequencies - harmonic)
        target = offset <= half_width
        reference = (offset > half_width) & (offset <= reference_width)
        if not target.any() or not reference.any():
            continue
        floor = np.median(magnitude[reference], axis=0)
        magnitude[target] = floor

    return np.fft.irfft(magnitude * phase, n=n_samples, axis=0)


def chain_magnitude(
    config: FilterChainConfig, sampling_rate: float, *, modality: Literal["emg", "mic"]
) -> Callable[[np.ndarray], np.ndarray]:
    """``|H(f)|`` of a whole chain, as actually applied.

    Squares the single-pass response when ``zero_phase`` is set, because forward-backward
    filtering applies it twice. Returning a callable rather than an array lets a caller --
    :func:`~bruxism.preprocessing.wavelets.assert_bands_within_passband`, the filter figure
    -- choose its own frequency grid.
    """
    stages = config.emg_stages if modality == "emg" else config.mic_stages

    def magnitude(frequencies: np.ndarray) -> np.ndarray:
        grid = np.asarray(frequencies, dtype=np.float64)
        response = np.ones_like(grid)
        for stage in stages:
            response = response * stage.magnitude_response(grid, sampling_rate)
        return response**2 if config.zero_phase else response

    return magnitude


def white_noise_power_gain(
    config: FilterChainConfig,
    sampling_rate: float,
    *,
    modality: Literal["emg", "mic"],
    n_points: int = 4096,
) -> float:
    """Fraction of a white input's power that survives this chain.

    Mean ``|H(f)|^2`` over the whole 0-Nyquist band, doubled for the zero-phase path
    because forward-backward filtering applies the response twice. This is exactly the
    factor that scales a quantisation floor: ADC dither is white, so of the ``step^2 / 12``
    variance it contributes, only this fraction reaches the analysis band.

    Computed from the designed response rather than assumed from the nominal passband
    edges, so a chain with seven notches in it is priced correctly.

    A ``spectral_interpolation`` stage has no LTI response; its *intended* rectangular
    suppression is used, which is what :meth:`FilterStage.magnitude_response` returns and
    what the filter figure draws.
    """
    grid = np.linspace(0.0, sampling_rate / 2.0, n_points)
    magnitude = chain_magnitude(config, sampling_rate, modality=modality)
    return float(np.mean(magnitude(grid) ** 2))


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
