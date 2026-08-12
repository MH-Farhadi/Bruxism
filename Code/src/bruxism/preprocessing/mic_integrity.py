"""Integrity measurements for the microphone channel.

This module exists because the interference audit that produced the corrected EMG filter
chain (``cause.md``, :mod:`bruxism.preprocessing.interference`) was run on the EMG columns
and never on the microphone column. Running the equivalent checks on the microphone, after
the modality ablation had already been reported, found a defect of a different kind. The
measurements are collected here so that every one of them is computed once, by name, and
stored in the manifest rather than rediscovered by hand.

What the audit found in the 2025-08 collection (see ``audio.md`` for the full write-up):

* **The channel is duplicated across participants.** Fingerprinting each channel by a
  rotation-invariant signature gives 100 distinct waveforms on each of the four EMG columns
  and **37** on the microphone column. 83 of the 100 recordings carry a microphone waveform
  that is bit-identical, after a circular rotation of 0.2-8 s, to another *participant's*
  recording of the same condition, and all four S01-S04 quiet-rest recordings share one
  waveform. Leave-one-subject-out therefore does not hold out the audio.
* **It is not aligned with the EMG.** Over the 45 chewing recordings the envelope
  cross-correlation at zero lag has a median of -0.017 and peaks at a median absolute lag
  of 18 s.
* **It is an envelope, not a waveform.** 96 % of its power lies below 10 Hz (range
  91.4-98.7 %), it is integer-valued with a 1.0-count step, and the 20 Hz high-pass of the
  production chain retains a median of 1.19 % of its variance. For the clenching and
  grinding conditions what survives is at or below the quantisation floor.

Each measurement below maps to exactly one
:class:`~bruxism.data.quality.QualityFlag`, and the thresholds are declared here rather
than at the call site so that a manifest can state which version of them it was built
under.

The functions take raw signals and return numbers. They never repair anything: a rotated
duplicate is *reported*, not de-rotated, because the rotation offset relative to the EMG is
unknown and a guessed re-alignment would be a fabrication presented as a fix.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from bruxism.utils.logging import get_logger

__all__ = [
    "ALIGNMENT_ENVELOPE_BAND_HZ",
    "ALIGNMENT_MAX_LAG_SECONDS",
    "ALIGNMENT_MIN_R_AT_ZERO",
    "ENVELOPE_BANDWIDTH_CEILING_HZ",
    "ENVELOPE_POWER_FRACTION_THRESHOLD",
    "MIC_INTEGRITY_POLICY_VERSION",
    "QUANTISATION_FLOOR_MARGIN_DB",
    "EnvelopeAlignment",
    "MicIntegrity",
    "confirm_rotations",
    "duplicate_groups",
    "is_circular_rotation",
    "measure_envelope_alignment",
    "measure_mic_integrity",
    "power_fraction_below",
    "quantisation_step",
    "summarise_duplication",
    "waveform_fingerprint",
]

logger = get_logger(__name__)

#: Bump when any threshold or measurement definition in this module changes. Recorded in
#: the manifest so a stored flag can always be traced to the rule that raised it.
MIC_INTEGRITY_POLICY_VERSION: Final[str] = "2026-08-12.1"

#: Frequency below which power is counted as "envelope rather than waveform". A sound-level
#: sensor's output is a rectified, smoothed envelope and lives almost entirely under a few
#: hertz; an acoustic waveform of tooth contact does not.
ENVELOPE_BANDWIDTH_CEILING_HZ: Final[float] = 10.0

#: Share of raw power below :data:`ENVELOPE_BANDWIDTH_CEILING_HZ` above which the channel is
#: reported as an envelope. Declared, not tuned: every recording in this dataset scores
#: 0.914-0.987 and a waveform channel would score a small fraction of that.
ENVELOPE_POWER_FRACTION_THRESHOLD: Final[float] = 0.90

#: A retained band within this many dB of the quantisation floor carries no measurable
#: signal. 3 dB is the point at which the signal and the dither contribute equal variance.
QUANTISATION_FLOOR_MARGIN_DB: Final[float] = 3.0

#: Band used to build the envelopes that the EMG/mic alignment is measured on. The lower
#: edge removes drift; the upper edge sits below the 600 Hz Nyquist of the 1200 Hz rate.
ALIGNMENT_ENVELOPE_BAND_HZ: Final[tuple[float, float]] = (20.0, 450.0)

#: Envelope smoothing cutoff. Chewing rhythm is 1-2 Hz, so 3 Hz keeps the burst structure
#: and removes the carrier.
_ALIGNMENT_SMOOTHING_HZ: Final[float] = 3.0

#: Largest absolute lag, in seconds, at which two channels of one recording are still
#: considered aligned. Generous: the transition guard is 0.25 s, so anything the guard can
#: absorb is not worth flagging.
ALIGNMENT_MAX_LAG_SECONDS: Final[float] = 0.25

#: Zero-lag envelope correlation below which two channels that should co-vary do not.
#: Chewing produces a muscle burst and a sound at the same instant; a correctly wired pair
#: scores well above this and the recordings here score -0.017.
ALIGNMENT_MIN_R_AT_ZERO: Final[float] = 0.10

#: Welch segment length, matching :mod:`bruxism.preprocessing.interference`.
_NPERSEG: Final[int] = 4096

#: Smallest variance treated as non-zero, so a dead channel divides safely.
_EPS: Final[float] = 1e-12


def waveform_fingerprint(signal: np.ndarray) -> str:
    """Rotation-invariant SHA-256 of a channel's samples.

    The digest is taken over the **sorted** samples, so it is invariant to any permutation
    of them and therefore in particular to a circular rotation. Two recordings that share a
    fingerprint are not merely similar: in this dataset every such pair turned out to be
    bit-identical after :func:`numpy.roll`, which :func:`is_circular_rotation` confirms.

    Sorting rather than correlating is what makes the check affordable as a manifest
    column: it is one pass per recording and a dictionary lookup afterwards, instead of the
    quadratic cross-correlation of every recording against every other.

    Parameters
    ----------
    signal
        1-D array of samples. Cast to ``float64`` first so that a column stored as ``int64``
        in one file and ``float64`` in another produces the same digest -- both occur in
        this dataset and the dtype is an artefact of CSV parsing, not of the signal.
    """
    values = np.ascontiguousarray(np.sort(np.asarray(signal, dtype=np.float64).ravel()))
    return hashlib.sha256(values.tobytes()).hexdigest()


def is_circular_rotation(a: np.ndarray, b: np.ndarray) -> tuple[bool, int]:
    """Whether ``b`` is ``a`` rotated, and by how many samples.

    Confirms what :func:`waveform_fingerprint` only suggests. The candidate offset comes
    from an FFT cross-correlation and the equality is then checked exactly, so a ``True``
    here means the two arrays contain the same samples in the same cyclic order -- not that
    they are merely well correlated.

    Returns
    -------
    (is_rotation, offset)
        ``offset`` is ``k`` such that ``np.roll(a, -k) == b``. It is ``-1`` when the arrays
        are not rotations of each other or have different lengths.
    """
    left = np.asarray(a, dtype=np.float64).ravel()
    right = np.asarray(b, dtype=np.float64).ravel()
    if left.size != right.size or left.size == 0:
        return False, -1
    spectrum = np.fft.rfft(left - left.mean()) * np.conj(np.fft.rfft(right - right.mean()))
    offset = int(np.argmax(np.fft.irfft(spectrum, n=left.size)))
    if np.array_equal(np.roll(left, -offset), right):
        return True, offset
    return False, -1


def quantisation_step(signal: np.ndarray) -> float:
    """Smallest non-zero difference between distinct sample values.

    For an integer-valued ADC channel this is the least significant bit, and
    ``step**2 / 12`` is the variance the quantiser itself contributes. Returns ``nan`` for a
    constant channel, which has no step to measure and is caught by the dead-channel flag
    instead.
    """
    unique = np.unique(np.asarray(signal, dtype=np.float64).ravel())
    if unique.size < 2:
        return float("nan")
    return float(np.min(np.diff(unique)))


def power_fraction_below(
    signal: np.ndarray, sampling_rate: float, ceiling_hz: float = ENVELOPE_BANDWIDTH_CEILING_HZ
) -> float:
    """Share of a channel's total power lying below ``ceiling_hz``.

    The mean is removed first, so a large DC offset -- which this channel has -- does not
    land in the lowest bin and guarantee the answer.
    """
    values = np.asarray(signal, dtype=np.float64).ravel()
    if values.size < 16:
        raise ValueError(f"need at least 16 samples to estimate a spectrum, got {values.size}")
    nperseg = int(min(_NPERSEG, values.size))
    frequencies, psd = welch(values - values.mean(), fs=sampling_rate, nperseg=nperseg)
    total = float(psd.sum())
    if total <= 0:
        return float("nan")
    return float(psd[frequencies < ceiling_hz].sum() / total)


@dataclass(frozen=True)
class EnvelopeAlignment:
    """How well two channels of one recording line up in time.

    Attributes
    ----------
    r_at_zero
        Envelope correlation with no shift applied. This is the number that matters: two
        correctly wired channels of the same event are aligned at lag 0 by construction.
    best_lag_seconds
        Shift at which the envelopes correlate best. Signed; positive means the second
        channel lags the first.
    peak_r
        Correlation at ``best_lag_seconds``. A low peak means the two channels do not
        describe the same event at *any* shift, which is a stronger failure than a
        misalignment.
    """

    r_at_zero: float
    best_lag_seconds: float
    peak_r: float

    @property
    def is_aligned(self) -> bool:
        """Whether the pair is aligned well enough to be fused window by window."""
        return (
            abs(self.best_lag_seconds) <= ALIGNMENT_MAX_LAG_SECONDS
            and self.r_at_zero >= ALIGNMENT_MIN_R_AT_ZERO
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "r_at_zero": self.r_at_zero,
            "best_lag_seconds": self.best_lag_seconds,
            "peak_r": self.peak_r,
        }


def _envelope(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Band-limited analytic envelope, smoothed to the burst-rhythm timescale."""
    values = np.asarray(signal, dtype=np.float64).ravel()
    values = values - values.mean()
    nyquist = sampling_rate / 2.0
    low, high = ALIGNMENT_ENVELOPE_BAND_HZ
    high = min(high, nyquist * 0.98)
    band = butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
    smooth = butter(2, _ALIGNMENT_SMOOTHING_HZ / nyquist, btype="low", output="sos")
    return sosfiltfilt(smooth, np.abs(hilbert(sosfiltfilt(band, values))))


def measure_envelope_alignment(
    reference: np.ndarray, other: np.ndarray, sampling_rate: float
) -> EnvelopeAlignment:
    """Cross-correlate two channels' envelopes to find their relative timing.

    Used to ask whether the microphone column of a recording describes the same events as
    its EMG columns. On a chewing recording the answer should be an emphatic yes: every
    chew is a muscle burst and a sound at the same instant.

    The correlation is circular (computed through the FFT), which is the right choice here
    because the failure mode being tested for is itself a circular rotation.
    """
    left = _envelope(reference, sampling_rate)
    right = _envelope(other, sampling_rate)
    left = left - left.mean()
    right = right - right.mean()
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    if norm <= _EPS:
        return EnvelopeAlignment(r_at_zero=float("nan"), best_lag_seconds=0.0, peak_r=float("nan"))
    correlation = np.fft.irfft(np.fft.rfft(left) * np.conj(np.fft.rfft(right)), n=left.size) / norm
    index = int(np.argmax(correlation))
    lag = index if index < left.size // 2 else index - left.size
    return EnvelopeAlignment(
        r_at_zero=float(correlation[0]),
        best_lag_seconds=float(lag / sampling_rate),
        peak_r=float(correlation[index]),
    )


@dataclass(frozen=True)
class MicIntegrity:
    """Everything measurable about one recording's microphone channel, in isolation.

    Cross-recording duplication is deliberately *not* here: it cannot be decided from one
    file. :func:`duplicate_groups` decides it from the fingerprints of the whole manifest.
    """

    fingerprint: str
    n_unique_values: int
    quantisation_step: float
    raw_variance: float
    #: Share of raw power below :data:`ENVELOPE_BANDWIDTH_CEILING_HZ`.
    power_fraction_below_10hz: float
    #: Variance after the analysis chain, as a fraction of the raw variance.
    variance_retained_fraction: float
    #: Post-chain variance in excess of the quantiser's own contribution, in dB.
    snr_above_quantisation_db: float
    alignment: EnvelopeAlignment

    @property
    def is_dead(self) -> bool:
        """Constant channel, or one whose whole variance is below the quantiser's."""
        if not np.isfinite(self.quantisation_step):
            return True
        floor = self.quantisation_step**2 / 12.0
        return self.raw_variance <= floor

    @property
    def is_envelope_not_waveform(self) -> bool:
        return (
            np.isfinite(self.power_fraction_below_10hz)
            and self.power_fraction_below_10hz > ENVELOPE_POWER_FRACTION_THRESHOLD
        )

    @property
    def is_at_quantisation_floor(self) -> bool:
        # -inf is the *worst* case, not an unmeasurable one: it means the post-chain
        # variance did not exceed the quantiser's own contribution at all. Only nan --
        # a constant channel with no step to measure -- is undecidable here, and the
        # dead-channel flag covers that.
        return not np.isnan(self.snr_above_quantisation_db) and (
            self.snr_above_quantisation_db < QUANTISATION_FLOOR_MARGIN_DB
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "n_unique_values": self.n_unique_values,
            "quantisation_step": self.quantisation_step,
            "raw_variance": self.raw_variance,
            "power_fraction_below_10hz": self.power_fraction_below_10hz,
            "variance_retained_fraction": self.variance_retained_fraction,
            "snr_above_quantisation_db": self.snr_above_quantisation_db,
            "alignment": self.alignment.to_dict(),
            "policy_version": MIC_INTEGRITY_POLICY_VERSION,
        }


def measure_mic_integrity(
    mic: np.ndarray,
    sampling_rate: float,
    *,
    filtered_mic: np.ndarray | None = None,
    retained_bandwidth_fraction: float = 1.0,
    emg: np.ndarray | None = None,
) -> MicIntegrity:
    """Measure one recording's microphone channel.

    Parameters
    ----------
    mic
        Raw microphone samples, before any offline filtering.
    filtered_mic
        The same channel after the analysis chain. When supplied, the retained-variance
        and quantisation-SNR figures describe the signal the model actually receives.
        When ``None`` both are computed against the raw channel, i.e. as if the chain were
        a pass-through.
    retained_bandwidth_fraction
        Share of the Nyquist band the chain passes. Quantisation noise is white, so only
        this fraction of ``step**2 / 12`` survives into the analysis band. Passing 1.0
        (the default) is the conservative choice: it overstates the floor slightly and so
        under-reports the SNR.
    emg
        ``(n_samples, n_channels)`` EMG for the alignment measurement. The channel with the
        largest variance is used as the reference, because electrode quality varies and a
        flat channel would produce a meaningless correlation. Alignment is reported as
        ``nan`` when no EMG is supplied.
    """
    raw = np.asarray(mic, dtype=np.float64).ravel()
    step = quantisation_step(raw)
    raw_variance = float(raw.var())

    analysed = raw if filtered_mic is None else np.asarray(filtered_mic, dtype=np.float64).ravel()
    retained = float(analysed.var() / raw_variance) if raw_variance > _EPS else float("nan")

    if np.isfinite(step):
        floor = (step**2 / 12.0) * float(retained_bandwidth_fraction)
        excess = float(analysed.var()) - floor
        snr_db = 10.0 * np.log10(excess / floor) if excess > 0 and floor > 0 else -np.inf
    else:
        snr_db = float("nan")

    if emg is None:
        alignment = EnvelopeAlignment(
            r_at_zero=float("nan"), best_lag_seconds=float("nan"), peak_r=float("nan")
        )
    else:
        channels = np.asarray(emg, dtype=np.float64)
        if channels.ndim == 1:
            channels = channels[:, None]
        reference = channels[:, int(np.argmax(channels.var(axis=0)))]
        alignment = measure_envelope_alignment(reference, raw, sampling_rate)

    return MicIntegrity(
        fingerprint=waveform_fingerprint(raw),
        n_unique_values=int(np.unique(raw).size),
        quantisation_step=step,
        raw_variance=raw_variance,
        power_fraction_below_10hz=power_fraction_below(raw, sampling_rate),
        variance_retained_fraction=retained,
        snr_above_quantisation_db=float(snr_db),
        alignment=alignment,
    )


def duplicate_groups(
    fingerprints: Mapping[str, str],
    *,
    subject_of: Mapping[str, str] | None = None,
    cross_subject_only: bool = True,
) -> dict[str, list[str]]:
    """Group recording ids that share a channel fingerprint.

    Parameters
    ----------
    fingerprints
        ``recording_id -> fingerprint``, as produced by :func:`waveform_fingerprint`.
    subject_of
        ``recording_id -> subject_id``. Required when ``cross_subject_only`` is set.
    cross_subject_only
        ``True`` returns only groups spanning more than one participant -- the ones that
        break leave-one-subject-out. A repeat within one participant is a different
        problem: it is still wrong, but it does not put held-out signal into the training
        set, so it is reported separately rather than mixed in.

    Returns
    -------
    dict
        ``fingerprint -> sorted recording ids``, for groups of size > 1 only.
    """
    grouped: dict[str, list[str]] = {}
    for recording_id, fingerprint in fingerprints.items():
        grouped.setdefault(fingerprint, []).append(recording_id)

    result: dict[str, list[str]] = {}
    for fingerprint, members in grouped.items():
        if len(members) < 2:
            continue
        if cross_subject_only:
            if subject_of is None:
                raise ValueError("cross_subject_only=True requires subject_of")
            if len({subject_of[member] for member in members}) < 2:
                continue
        result[fingerprint] = sorted(members)
    return result


def summarise_duplication(
    fingerprints: Mapping[str, str], subject_of: Mapping[str, str]
) -> dict[str, Any]:
    """One-glance summary of a channel's duplication, for the audit report and the tests."""
    cross = duplicate_groups(fingerprints, subject_of=subject_of, cross_subject_only=True)
    within = duplicate_groups(fingerprints, subject_of=subject_of, cross_subject_only=False)
    affected = sorted({member for members in cross.values() for member in members})
    per_subject: dict[str, int] = {}
    for recording_id in affected:
        subject = subject_of[recording_id]
        per_subject[subject] = per_subject.get(subject, 0) + 1
    return {
        "n_recordings": len(fingerprints),
        "n_distinct_waveforms": len(set(fingerprints.values())),
        "n_cross_subject_groups": len(cross),
        "n_within_subject_only_groups": len(within) - len(cross),
        "n_recordings_affected": len(affected),
        "affected_per_subject": dict(sorted(per_subject.items())),
        "groups": {fingerprint[:8]: members for fingerprint, members in sorted(cross.items())},
    }


def confirm_rotations(
    signals: Mapping[str, np.ndarray], group: Sequence[str] | Iterable[str]
) -> dict[str, int]:
    """Confirm every member of a fingerprint group is a rotation of the first, exactly.

    Returns ``recording_id -> offset``. A member that shares the fingerprint but is *not* a
    rotation gets ``-1``; that would mean two different recordings happen to contain the
    same multiset of samples, which is worth knowing about separately.
    """
    members = list(group)
    if not members:
        return {}
    reference = signals[members[0]]
    offsets: dict[str, int] = {members[0]: 0}
    for member in members[1:]:
        _, offset = is_circular_rotation(reference, signals[member])
        offsets[member] = offset
    return offsets
