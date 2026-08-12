"""Deterministic discovery and validation of the raw dataset.

The manifest is the single source of truth about what was recorded. It is built by
scanning every ``*.csv`` under the data root -- including the secondary
``Data/More Data/Data/Subject_5/`` location -- in a sorted, directory-order-independent
way, reading each file once, and recording checksums, trigger structure, signal ranges,
metadata agreement and quality flags.

**Two passes, not one.** Every check in :func:`_iter_records` looks at a single file in
isolation, and for four years of this project that was the whole manifest. It is not
enough: the microphone defect documented in ``audio.md`` is invisible from any one file and
obvious the moment two are compared. :func:`build_manifest` therefore runs a second,
cross-recording pass that groups recordings by a rotation-invariant fingerprint of each
channel and flags any waveform shared between participants. That pass is what turns
"83 recordings look fine individually" into "83 recordings carry another participant's
audio".

Privacy: the manifest stores canonical subject IDs (``S01``), data-root-relative POSIX
paths and numeric summaries only. It never stores participant names, absolute paths, or
any reference to surveys, photographs, receipts or reimbursement files.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from bruxism.data.labels import RAW_TOKEN_TO_CONDITION, TaskFamily, family_of_token
from bruxism.data.quality import (
    QUALITY_POLICY_VERSION,
    ExclusionPolicy,
    QualityFlag,
)
from bruxism.data.schema import (
    CANONICAL_COLUMNS,
    EMG_COLUMNS,
    MIC_COLUMN,
    NOMINAL_SAMPLING_RATE_HZ,
    TRIGGER_COLUMN,
    RecordingKey,
    SchemaError,
    parse_recording_filename,
    read_recording_csv,
    validate_columns,
)
from bruxism.preprocessing.filters import (
    FilterChainConfig,
    apply_filter_chain,
    white_noise_power_gain,
)
from bruxism.preprocessing.interference import (
    MAINS_CONTAMINATION_THRESHOLD,
    measure_mains_contamination,
)
from bruxism.preprocessing.mic_integrity import (
    MIC_INTEGRITY_POLICY_VERSION,
    duplicate_groups,
    measure_mic_integrity,
    waveform_fingerprint,
)
from bruxism.utils import progress
from bruxism.utils.io import file_sha256, hash_mapping
from bruxism.utils.logging import get_logger

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "DatasetManifest",
    "RecordingRecord",
    "TriggerRun",
    "build_manifest",
    "discover_recordings",
    "load_manifest",
    "trigger_runs",
]

logger = get_logger(__name__)

#: Bump when the manifest column set changes. Runs record the version they were built with.
#: 1.1 adds the mains-contamination columns.
#: 1.2 adds the microphone-integrity columns and the cross-recording duplication pass.
MANIFEST_SCHEMA_VERSION: Final[str] = "1.2"

#: Condition families in which a muscle burst and a sound are simultaneous by construction,
#: so a microphone that does not co-vary with the EMG is provably mis-wired rather than
#: merely quiet. Chewing is the only unambiguous case: every chew is an impact. Clenching
#: and grinding do make sound, but far less of it, and a null correlation there could be
#: honest silence.
_SOUND_EXPECTED_FAMILIES: Final[frozenset[str]] = frozenset({str(TaskFamily.CHEWING)})

#: Directories under the data root that are never scanned for recordings. They hold
#: administrative or identifiable material that is not research data.
_EXCLUDED_DIR_NAMES: Final[frozenset[str]] = frozenset({"Pics", "__MACOSX", ".git"})

#: Files that must never be read, hashed, copied or referenced by any generated artifact.
_PRIVATE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".pdf", ".xlsx", ".xls", ".heic", ".jpg", ".jpeg", ".png", ".doc", ".docx"}
)

_SUBJECT_DIR_PATTERN: Final[re.Pattern[str]] = re.compile(r"^Subject_(\d+)$")


@dataclass(frozen=True)
class TriggerRun:
    """One contiguous interval during which the trigger was high.

    Boundaries are half-open sample indices ``[start_sample, end_sample)`` relative to the
    start of the recording.
    """

    index: int
    start_sample: int
    end_sample: int

    @property
    def n_samples(self) -> int:
        return self.end_sample - self.start_sample

    def duration_seconds(self, sampling_rate: float) -> float:
        return self.n_samples / sampling_rate

    def to_dict(self) -> dict[str, int]:
        return {
            "index": self.index,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
        }


def trigger_runs(trigger: np.ndarray) -> list[TriggerRun]:
    """Split a binary trigger channel into its contiguous high runs.

    Parameters
    ----------
    trigger
        1-D array whose non-zero entries mark the active state.

    Returns
    -------
    list[TriggerRun]
        Runs in ascending sample order. Empty when the trigger never goes high.
    """
    binary = (np.asarray(trigger).ravel() > 0).astype(np.int8)
    if binary.size == 0:
        return []
    padded = np.concatenate([[0], binary, [0]])
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [
        TriggerRun(index=i, start_sample=int(s), end_sample=int(e))
        for i, (s, e) in enumerate(zip(starts, ends))
    ]


def _as_float(value: str | None) -> float | None:
    """Parse a metadata value as a float, returning None when it is absent or malformed."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_metadata(path: Path) -> dict[str, str]:
    """Parse the acquisition ``key: value`` metadata sidecar."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        values[key.strip()] = value.strip()
    return values


def _probe_video(path: Path) -> dict[str, Any]:
    """Read frame count / rate / size / codec from a video container.

    Returns a dict whose values are ``None`` when the video cannot be probed; a missing
    video never blocks manifest construction because video is not a model input.
    """
    result: dict[str, Any] = {
        "video_frame_count": None,
        "video_fps": None,
        "video_duration_seconds": None,
        "video_width": None,
        "video_height": None,
        "video_codec": None,
        "video_readable": False,
    }
    try:
        import cv2
    except ImportError:
        logger.debug("opencv not installed; skipping video probe for %s", path.name)
        return result

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return result
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        result.update(
            video_frame_count=frames if frames > 0 else None,
            video_fps=fps if fps > 0 else None,
            video_duration_seconds=(frames / fps) if frames > 0 and fps > 0 else None,
            video_width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
            video_height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
            video_codec="".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip(),
            video_readable=True,
        )
    finally:
        capture.release()
    return result


def _npy_agreement(npy_path: Path, frame: pd.DataFrame) -> tuple[bool, str]:
    """Check whether an ``.npy`` companion reproduces the CSV.

    Returns ``(agrees, detail)``; ``detail`` names the first disagreement found.
    """
    try:
        array = np.load(npy_path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - any load failure means "do not trust it"
        return False, f"unreadable: {exc}"
    if array.ndim != 2 or array.shape[1] != len(CANONICAL_COLUMNS):
        return False, f"shape {array.shape} does not match CSV schema"
    if array.shape[0] != len(frame):
        return False, f"row count {array.shape[0]} != CSV {len(frame)}"
    csv_values = frame[list(CANONICAL_COLUMNS)].to_numpy(dtype=np.float64)
    for column_index, column in enumerate(CANONICAL_COLUMNS):
        if not np.allclose(array[:, column_index], csv_values[:, column_index], atol=1e-3):
            return False, f"column {column!r} differs from the CSV"
    return True, ""


def _companion_index(data_root: Path) -> dict[str, Path]:
    """Map every companion file name under the data root to its physical path.

    Companions do not always sit beside their CSV: S05's protrusion metadata lives in
    ``Data/More Data/Data/Subject_5/`` while its CSV is in the primary directory. Resolving
    by file name across the whole root finds those without moving any file.
    """
    index: dict[str, Path] = {}
    for suffix in (".avi", ".npy", ".txt"):
        for path in sorted(data_root.rglob(f"*{suffix}")):
            if any(part in _EXCLUDED_DIR_NAMES for part in path.relative_to(data_root).parts):
                continue
            index.setdefault(path.name, path)
    return index


def _resolve_companion(
    csv_path: Path, suffix: str, companions: dict[str, Path], *, metadata: bool = False
) -> Path | None:
    """Find a CSV's companion beside it, or elsewhere under the data root by file name."""
    name = f"{csv_path.stem}_metadata.txt" if metadata else f"{csv_path.stem}{suffix}"
    local = csv_path.with_name(name)
    if local.is_file():
        return local
    return companions.get(name)


def discover_recordings(data_root: Path | str) -> list[tuple[RecordingKey, Path]]:
    """Find every recording CSV under ``data_root``, deterministically.

    Recursively walks the data root, skipping administrative and image directories, and
    returns ``(key, csv_path)`` pairs sorted by recording id. Directory iteration order
    never affects the result.

    Raises
    ------
    FileNotFoundError
        If ``data_root`` does not exist.
    SchemaError
        If a CSV filename does not match the expected pattern.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root not found or not a directory: {root}")

    found: list[tuple[RecordingKey, Path]] = []
    for csv_path in sorted(root.rglob("*.csv")):
        if any(part in _EXCLUDED_DIR_NAMES for part in csv_path.relative_to(root).parts):
            continue
        # Only files that sit inside a Subject_<n> directory are recordings.
        if not any(_SUBJECT_DIR_PATTERN.match(part) for part in csv_path.parts):
            logger.warning("skipping CSV outside any Subject_<n> directory: %s", csv_path.name)
            continue
        found.append((parse_recording_filename(csv_path.name), csv_path))

    found.sort(key=lambda item: (item[0].recording_id, item[1].as_posix()))
    duplicates = _duplicate_ids(key for key, _ in found)
    if duplicates:
        raise SchemaError(
            f"duplicate recording ids discovered (same subject/condition/timestamp in two "
            f"locations): {sorted(duplicates)}"
        )
    return found


def _duplicate_ids(keys: Iterable[RecordingKey]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for key in keys:
        if key.recording_id in seen:
            duplicates.add(key.recording_id)
        seen.add(key.recording_id)
    return duplicates


@dataclass
class RecordingRecord:
    """One manifest row: everything known about a single recording."""

    recording_id: str
    subject_id: str
    condition_token: str
    condition: str
    task_family: str
    repetition_token: str
    csv_relpath: str
    avi_relpath: str | None
    metadata_relpath: str | None
    n_samples: int
    sampling_rate_hz: int
    metadata_sampling_rate_hz: int | None
    duration_seconds: float
    columns: tuple[str, ...]
    trigger_values: tuple[float, ...]
    trigger_active_samples: int
    trigger_active_fraction: float
    n_trigger_runs: int
    n_trigger_transitions: int
    trigger_run_boundaries: list[dict[str, int]]
    emg_min: list[float]
    emg_max: list[float]
    mic_min: float
    mic_max: float
    csv_sha256: str
    avi_sha256: str | None
    metadata_sha256: str | None
    metadata_condition_key: str | None
    metadata_status: str | None
    metadata_samples_saved: int | None
    metadata_target_duration_seconds: float | None
    npy_relpath: str | None
    npy_claimed: bool
    npy_agrees_with_csv: bool | None
    npy_disagreement: str | None
    video_frame_count: int | None
    video_fps: float | None
    video_duration_seconds: float | None
    video_width: int | None
    video_height: int | None
    video_codec: str | None
    video_readable: bool
    startup_transient_seconds: float
    startup_transient_peak_ratio: float
    #: Fraction of the RAW 20-450 Hz EMG power within 3 Hz of a mains multiple, pooled
    #: over channels. Measured before any offline filtering, so it describes the
    #: acquisition rather than the analysis.
    mains_harmonic_power_fraction: float
    #: Same fraction per EMG channel, in column order.
    mains_harmonic_power_fraction_per_channel: list[float]
    #: Per-harmonic breakdown, ``"60,0.0012;120,0.0033;..."`` in ascending frequency.
    mains_harmonic_breakdown: str
    quality_flags: list[str]
    excluded: bool
    exclusion_reason: str
    quality_policy_version: str
    manifest_schema_version: str = MANIFEST_SCHEMA_VERSION
    conflict_rules_applied: list[str] = field(default_factory=list)

    # --- microphone integrity (audio.md; schema 1.2) --------------------------------
    #: Rotation-invariant SHA-256 of the microphone samples. Two recordings sharing this
    #: contain the same waveform. This single column is what makes the duplication in
    #: ``audio.md`` 1.1 impossible to miss: 100 recordings, 37 distinct values.
    mic_sorted_sha256: str = ""
    #: Same fingerprint per EMG channel, in column order. Stored so the microphone finding
    #: can always be checked against a channel known to be clean -- all four are distinct
    #: across all 100 recordings.
    emg_sorted_sha256: list[str] = field(default_factory=list)
    #: Fingerprint of the trigger channel. Degenerate by design in some recordings (an
    #: all-zero rest trigger), which is why the duplication pass allows it explicitly.
    trigger_sorted_sha256: str = ""
    #: Set by the cross-recording pass: the shared fingerprint's short id, or ``""``.
    mic_duplicate_group: str = ""
    #: Other recordings carrying this microphone waveform, semicolon-separated.
    mic_duplicate_of: str = ""
    mic_n_unique_values: int = 0
    #: Least significant bit of the microphone channel, in ADC counts.
    mic_quantisation_step: float = float("nan")
    #: Share of raw microphone power below 10 Hz. An envelope output scores near 1.
    mic_power_fraction_below_10hz: float = float("nan")
    #: Variance after the microphone chain, as a fraction of the raw variance. Measured
    #: over the WHOLE recording, because the manifest predates windowing and must not
    #: depend on a segmentation policy.
    mic_variance_retained_fraction: float = float("nan")
    #: Post-chain variance in excess of the quantiser's own contribution, in dB. Also
    #: whole-recording, which is the conservative direction: pooling active and inactive
    #: intervals inflates the variance, so a recording that clears the floor here may still
    #: be at it inside its trigger-active windows. On this dataset the whole-recording rule
    #: flags 34 of 100 recordings and the trigger-active measurement flags 42.
    mic_snr_above_quantisation_db: float = float("nan")
    #: Envelope correlation between the microphone and the largest-variance EMG channel,
    #: with no shift applied. Two correctly wired channels of one event align at lag 0.
    mic_emg_envelope_r_at_zero: float = float("nan")
    #: Shift at which those envelopes correlate best, in seconds.
    mic_emg_envelope_lag_seconds: float = float("nan")
    #: Correlation at that best shift.
    mic_emg_envelope_peak_r: float = float("nan")
    #: Mains-harmonic share of the RAW microphone power. The audit that produced the
    #: corrected EMG chain was never run on this channel; now it always is.
    mic_mains_harmonic_power_fraction: float = float("nan")
    #: Which microphone chain the retained-variance and SNR columns describe.
    mic_chain_description: str = ""
    mic_integrity_policy_version: str = MIC_INTEGRITY_POLICY_VERSION

    def to_row(self) -> dict[str, Any]:
        """Flatten to a manifest row (list/tuple columns become JSON-ish strings)."""
        row = dict(self.__dict__)
        row["columns"] = ",".join(self.columns)
        row["trigger_values"] = ",".join(f"{value:g}" for value in self.trigger_values)
        row["quality_flags"] = ",".join(self.quality_flags)
        row["conflict_rules_applied"] = ",".join(self.conflict_rules_applied)
        row["emg_min"] = ",".join(f"{value:.6g}" for value in self.emg_min)
        row["emg_max"] = ",".join(f"{value:.6g}" for value in self.emg_max)
        row["emg_sorted_sha256"] = ",".join(self.emg_sorted_sha256)
        row["mains_harmonic_power_fraction_per_channel"] = ",".join(
            f"{value:.6g}" for value in self.mains_harmonic_power_fraction_per_channel
        )
        row["trigger_run_boundaries"] = ";".join(
            f"{run['start_sample']}-{run['end_sample']}" for run in self.trigger_run_boundaries
        )
        return row


@dataclass
class DatasetManifest:
    """A validated, hashable description of every discovered recording."""

    records: list[RecordingRecord]
    data_root: Path
    sampling_rate_hz: int
    policy: ExclusionPolicy
    manifest_hash: str
    #: Per-channel cross-recording duplication summary from :func:`flag_shared_waveforms`.
    #: Empty when the manifest was rehydrated from disk, where it is recomputed on demand.
    duplication: dict[str, Any] = field(default_factory=dict)

    @property
    def frame(self) -> pd.DataFrame:
        """Manifest as a flat DataFrame, sorted by recording id."""
        return pd.DataFrame([record.to_row() for record in self.records]).sort_values(
            "recording_id", ignore_index=True
        )

    @property
    def included(self) -> list[RecordingRecord]:
        """Records that survive the exclusion policy."""
        return [record for record in self.records if not record.excluded]

    @property
    def subject_ids(self) -> list[str]:
        return sorted({record.subject_id for record in self.included})

    def runs_for(self, recording_id: str) -> list[TriggerRun]:
        record = self.by_id(recording_id)
        return [
            TriggerRun(index=i, start_sample=b["start_sample"], end_sample=b["end_sample"])
            for i, b in enumerate(record.trigger_run_boundaries)
        ]

    def audio_blocking_flags(self) -> dict[str, tuple[str, ...]]:
        """Included recordings whose microphone channel is unusable, and why.

        Keyed by recording id; empty when every included recording's microphone may be
        consumed. Read by :func:`bruxism.runner.assert_modality_is_supported_by_data`
        before any run that sets ``modality`` to ``fusion`` or ``audio_only``.
        """
        blocked: dict[str, tuple[str, ...]] = {}
        for record in self.included:
            flags = self.policy.audio_blocked_by(
                frozenset(QualityFlag(value) for value in record.quality_flags)
            )
            if flags:
                blocked[record.recording_id] = tuple(flag.value for flag in flags)
        return blocked

    def by_id(self, recording_id: str) -> RecordingRecord:
        for record in self.records:
            if record.recording_id == recording_id:
                return record
        raise KeyError(f"recording {recording_id!r} is not in this manifest")

    def csv_path(self, record: RecordingRecord) -> Path:
        """Absolute path of a record's CSV, resolved against this manifest's data root."""
        return self.data_root / record.csv_relpath

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "quality_policy_version": self.policy.policy_version,
            "manifest_hash": self.manifest_hash,
            "sampling_rate_hz": self.sampling_rate_hz,
            "n_records": len(self.records),
            "n_included": len(self.included),
            "subject_ids": self.subject_ids,
            "records": [record.to_row() for record in self.records],
        }


def _summarise_for_hash(records: Sequence[RecordingRecord]) -> dict[str, Any]:
    """The subset of manifest content that defines identity.

    Deliberately excludes anything environment-specific (absolute paths, probe results
    that depend on installed codecs) so that the same data produces the same hash on any
    machine.
    """
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "policy_version": QUALITY_POLICY_VERSION,
        "records": sorted(
            (
                {
                    "recording_id": record.recording_id,
                    "csv_sha256": record.csv_sha256,
                    "n_samples": record.n_samples,
                    "condition": record.condition,
                    "excluded": record.excluded,
                    "quality_flags": sorted(record.quality_flags),
                    # Rounded: the fraction is a float derived from a Welch estimate, and
                    # the manifest hash must not depend on the last bit of a PSD.
                    "mains_harmonic_power_fraction": round(record.mains_harmonic_power_fraction, 6),
                    # Channel identity. Exact, not rounded: these are digests, and a
                    # dataset whose microphone content changes must not keep its hash.
                    "mic_sorted_sha256": record.mic_sorted_sha256,
                    "emg_sorted_sha256": list(record.emg_sorted_sha256),
                    "mic_duplicate_group": record.mic_duplicate_group,
                }
                for record in records
            ),
            key=lambda item: str(item["recording_id"]),
        ),
    }


def _measure_startup_transient(
    emg: np.ndarray, sampling_rate: int, policy: ExclusionPolicy
) -> tuple[float, float]:
    """Measure the amplifier-settling transient at the start of a recording.

    A robust scale is computed from the recording *after* the probe interval, so the
    transient cannot inflate its own threshold. Two numbers are returned, and both are
    stored in the manifest so the segmentation startup guard can be justified from data
    rather than from the prototype's unexplained blanket 3 s skip.

    Returns
    -------
    (settle_seconds, peak_ratio)
        ``settle_seconds`` is the end of the *initial contiguous* excursion above
        ``startup_transient_sigma`` times the robust scale -- i.e. how long the signal
        takes to settle, measured from sample 0. It is 0.0 when the recording starts
        already settled. ``peak_ratio`` is the largest excursion seen within the probe
        interval, in units of the robust scale.
    """
    probe = int(policy.startup_probe_seconds * sampling_rate)
    if emg.shape[0] <= probe * 2:
        return 0.0, 0.0
    tail = emg[probe:]
    centre = np.median(tail, axis=0)
    scale = np.median(np.abs(tail - centre), axis=0)
    scale = np.where(scale > 0, scale, np.finfo(np.float64).eps)

    head_ratio = np.abs(emg[:probe] - centre) / scale
    peak_ratio = float(head_ratio.max())

    # Collapse channels: a sample is "unsettled" if any channel exceeds the threshold.
    unsettled = (head_ratio > policy.startup_transient_sigma).any(axis=1)
    if not unsettled[0]:
        return 0.0, peak_ratio
    settled_at = np.flatnonzero(~unsettled)
    end = int(settled_at[0]) if settled_at.size else probe
    return end / sampling_rate, peak_ratio


def _iter_records(
    discovered: Sequence[tuple[RecordingKey, Path]],
    data_root: Path,
    sampling_rate: int,
    policy: ExclusionPolicy,
    *,
    probe_video: bool,
    hash_video: bool,
    filter_config: FilterChainConfig,
) -> Iterator[RecordingRecord]:
    companions = _companion_index(data_root)
    for key, csv_path in discovered:
        relative = csv_path.relative_to(data_root)
        flags: set[QualityFlag] = set()
        rules: list[str] = []

        frame = read_recording_csv(csv_path)
        validate_columns(frame.columns)

        emg = frame[list(EMG_COLUMNS)].to_numpy(dtype=np.float64)
        trigger = frame[TRIGGER_COLUMN].to_numpy(dtype=np.float64)
        mic = frame[MIC_COLUMN].to_numpy(dtype=np.float64)
        n_samples = len(frame)
        duration = n_samples / sampling_rate

        runs = trigger_runs(trigger)
        active_samples = int((trigger > 0).sum())

        condition = RAW_TOKEN_TO_CONDITION[key.condition_token]
        family = family_of_token(key.condition_token)

        # --- companions ------------------------------------------------------------
        avi_path = _resolve_companion(csv_path, ".avi", companions)
        metadata_path = _resolve_companion(csv_path, ".txt", companions, metadata=True)
        npy_path = _resolve_companion(csv_path, ".npy", companions)
        if avi_path is None or metadata_path is None:
            flags.add(QualityFlag.INCOMPLETE_TRIPLE)
        # A companion found outside the CSV's own directory is recorded, not moved.
        for companion in (avi_path, metadata_path, npy_path):
            if companion is not None and companion.parent != csv_path.parent:
                flags.add(QualityFlag.SECONDARY_LOCATION)

        metadata: dict[str, str] = _parse_metadata(metadata_path) if metadata_path else {}
        metadata_rate = (
            int(metadata["sampling_rate"]) if metadata.get("sampling_rate", "").isdigit() else None
        )
        metadata_samples = (
            int(metadata["samples_saved"]) if metadata.get("samples_saved", "").isdigit() else None
        )
        target_duration = _as_float(metadata.get("target_duration_seconds"))
        metadata_condition_key = metadata.get("condition_key")
        metadata_status = metadata.get("status")

        if metadata_rate is not None and metadata_rate != sampling_rate:
            flags.add(QualityFlag.SAMPLING_RATE_MISMATCH)
        if metadata_samples is not None and metadata_samples != n_samples:
            flags.add(QualityFlag.SAMPLE_COUNT_MISMATCH)
            rules.append("R3_csv_row_count_is_authoritative")
        if metadata_status is not None and metadata_status.upper() != "COMPLETED":
            flags.add(QualityFlag.NOT_COMPLETED)
        if metadata_condition_key is not None and metadata_condition_key != key.condition_token:
            flags.add(QualityFlag.METADATA_CONDITION_CONFLICT)
            rules.append("R1_filename_wins_for_condition")

        npy_claimed = bool(metadata.get("npy_file"))
        npy_agrees: bool | None = None
        npy_detail: str | None = None
        if npy_path is not None:
            npy_agrees, npy_detail = _npy_agreement(npy_path, frame)
            if not npy_agrees:
                flags.add(QualityFlag.STALE_NPY_COMPANION)
            rules.append("R2_csv_is_authoritative_over_npy")
        elif npy_claimed:
            flags.add(QualityFlag.MISSING_NPY_COMPANION)

        # --- signal / trigger quality -----------------------------------------------
        # "Short" is relative to the duration the acquisition software itself declared,
        # not a hard-coded 60 s: the threshold must follow the protocol, not the other way
        # round. With no declared target the flag cannot be evaluated and is not raised.
        if (
            target_duration is not None
            and duration < policy.short_recording_fraction * target_duration
        ):
            flags.add(QualityFlag.SHORT_RECORDING)
        if "More Data" in relative.parts:
            flags.add(QualityFlag.SECONDARY_LOCATION)
        settle_seconds, peak_ratio = _measure_startup_transient(emg, sampling_rate, policy)
        if settle_seconds > 0.0:
            flags.add(QualityFlag.STARTUP_TRANSIENT)

        # Measured on the RAW signal: the point is to describe what the amplifier
        # delivered, independently of whatever the offline chain later removes. A chain
        # that removes the interference must not make the interference invisible.
        mains = measure_mains_contamination(emg, sampling_rate)
        if mains.fraction > MAINS_CONTAMINATION_THRESHOLD:
            flags.add(QualityFlag.MAINS_CONTAMINATION)

        # --- microphone integrity ------------------------------------------------------
        # The same measurement, on the channel it was never run on. Everything here is
        # per-recording; the duplication check needs all of them and lives in
        # build_manifest().
        mic_mains = measure_mains_contamination(mic, sampling_rate)
        filtered_mic = apply_filter_chain(mic, filter_config, sampling_rate, modality="mic")
        integrity = measure_mic_integrity(
            mic,
            sampling_rate,
            filtered_mic=filtered_mic,
            retained_bandwidth_fraction=white_noise_power_gain(
                filter_config, sampling_rate, modality="mic"
            ),
            emg=emg,
        )
        if integrity.is_dead:
            flags.add(QualityFlag.MIC_CHANNEL_DEAD)
        if integrity.is_envelope_not_waveform:
            flags.add(QualityFlag.MIC_BANDWIDTH_IMPLAUSIBLE)
        if integrity.is_at_quantisation_floor:
            flags.add(QualityFlag.MIC_AT_QUANTISATION_FLOOR)
        if str(family) in _SOUND_EXPECTED_FAMILIES and not integrity.alignment.is_aligned:
            flags.add(QualityFlag.MIC_EMG_UNALIGNED)

        is_rest = condition == "rest"
        if is_rest and runs:
            flags.add(QualityFlag.UNEXPECTED_TRIGGER_IN_REST)
        if not is_rest and not runs:
            flags.add(QualityFlag.NO_TRIGGER_ACTIVITY)

        video_info: dict[str, Any] = {
            "video_frame_count": None,
            "video_fps": None,
            "video_duration_seconds": None,
            "video_width": None,
            "video_height": None,
            "video_codec": None,
            "video_readable": False,
        }
        if probe_video and avi_path is not None:
            video_info = _probe_video(avi_path)
            if not video_info["video_readable"]:
                flags.add(QualityFlag.VIDEO_UNREADABLE)
            else:
                video_duration = video_info["video_duration_seconds"]
                if (
                    video_duration is not None
                    and abs(video_duration - duration) > policy.video_duration_tolerance_seconds
                ):
                    flags.add(QualityFlag.VIDEO_DURATION_MISMATCH)

        excluded, reason = policy.evaluate(frozenset(flags), duration)

        yield RecordingRecord(
            recording_id=key.recording_id,
            subject_id=key.subject_id,
            condition_token=key.condition_token,
            condition=condition,
            task_family=str(family),
            repetition_token=key.repetition_token,
            csv_relpath=relative.as_posix(),
            avi_relpath=(
                avi_path.relative_to(data_root).as_posix() if avi_path is not None else None
            ),
            metadata_relpath=(
                metadata_path.relative_to(data_root).as_posix()
                if metadata_path is not None
                else None
            ),
            n_samples=n_samples,
            sampling_rate_hz=sampling_rate,
            metadata_sampling_rate_hz=metadata_rate,
            duration_seconds=duration,
            columns=tuple(str(column) for column in frame.columns),
            trigger_values=tuple(float(value) for value in np.unique(trigger)),
            trigger_active_samples=active_samples,
            trigger_active_fraction=active_samples / n_samples if n_samples else 0.0,
            n_trigger_runs=len(runs),
            n_trigger_transitions=2 * len(runs),
            trigger_run_boundaries=[run.to_dict() for run in runs],
            emg_min=[float(value) for value in emg.min(axis=0)],
            emg_max=[float(value) for value in emg.max(axis=0)],
            mic_min=float(mic.min()),
            mic_max=float(mic.max()),
            csv_sha256=file_sha256(csv_path),
            avi_sha256=(file_sha256(avi_path) if (hash_video and avi_path is not None) else None),
            metadata_sha256=(file_sha256(metadata_path) if metadata_path is not None else None),
            metadata_condition_key=metadata_condition_key,
            metadata_status=metadata_status,
            metadata_samples_saved=metadata_samples,
            metadata_target_duration_seconds=target_duration,
            npy_relpath=(
                npy_path.relative_to(data_root).as_posix() if npy_path is not None else None
            ),
            npy_claimed=npy_claimed,
            npy_agrees_with_csv=npy_agrees,
            npy_disagreement=npy_detail or None,
            startup_transient_seconds=settle_seconds,
            startup_transient_peak_ratio=peak_ratio,
            mains_harmonic_power_fraction=mains.fraction,
            mains_harmonic_power_fraction_per_channel=list(mains.per_channel),
            mains_harmonic_breakdown=";".join(
                f"{frequency},{value:.6g}" for frequency, value in mains.per_harmonic.items()
            ),
            mic_sorted_sha256=integrity.fingerprint,
            emg_sorted_sha256=[waveform_fingerprint(emg[:, i]) for i in range(emg.shape[1])],
            trigger_sorted_sha256=waveform_fingerprint(trigger),
            # Filled by the cross-recording pass in build_manifest(); a single file cannot
            # know whether its waveform appears anywhere else.
            mic_duplicate_group="",
            mic_duplicate_of="",
            mic_n_unique_values=integrity.n_unique_values,
            mic_quantisation_step=integrity.quantisation_step,
            mic_power_fraction_below_10hz=integrity.power_fraction_below_10hz,
            mic_variance_retained_fraction=integrity.variance_retained_fraction,
            mic_snr_above_quantisation_db=integrity.snr_above_quantisation_db,
            mic_emg_envelope_r_at_zero=integrity.alignment.r_at_zero,
            mic_emg_envelope_lag_seconds=integrity.alignment.best_lag_seconds,
            mic_emg_envelope_peak_r=integrity.alignment.peak_r,
            mic_mains_harmonic_power_fraction=mic_mains.fraction,
            mic_chain_description=" -> ".join(
                stage.describe() for stage in filter_config.mic_stages
            ),
            mic_integrity_policy_version=MIC_INTEGRITY_POLICY_VERSION,
            quality_flags=sorted(flag.value for flag in flags),
            excluded=excluded,
            exclusion_reason=reason,
            quality_policy_version=policy.policy_version,
            conflict_rules_applied=sorted(set(rules)),
            **video_info,
        )


def flag_shared_waveforms(records: Sequence[RecordingRecord]) -> dict[str, Any]:
    """Second pass: find channel waveforms shared between participants, and flag them.

    Mutates ``records`` in place, setting :attr:`RecordingRecord.mic_duplicate_group`,
    :attr:`RecordingRecord.mic_duplicate_of` and the
    :data:`~bruxism.data.quality.QualityFlag.MIC_WAVEFORM_DUPLICATED` flag on every
    recording whose microphone waveform also appears under a different subject.

    Exclusion is deliberately **not** re-evaluated. The flag is not an excluding flag: the
    EMG of these recordings is sound and removing 83 of 100 files would delete the dataset
    to protect a channel the EMG results never read. See rule
    ``R4_mic_channel_is_not_analysable_audio`` and
    :attr:`~bruxism.data.quality.ExclusionPolicy.audio_blocking_flags`.

    The EMG and trigger channels are checked too, and their result is returned rather than
    flagged. EMG duplication would be a far more serious finding than the microphone one --
    it would invalidate every result in the project -- so it is measured every time the
    manifest is built rather than assumed absent.

    **The trigger result is informational and must not be read as a defect.** The
    fingerprint is rotation-invariant because it is taken over sorted samples, and for a
    *binary* channel that means it collides whenever two recordings merely share a duty
    cycle: an all-zero rest trigger matches every other all-zero trigger, and a scripted
    protocol that gives every participant the same number of active samples matches
    trivially. Those collisions say nothing about whether a signal was copied. The
    measured channels -- the four EMG columns and the microphone -- carry continuous
    values, so a shared fingerprint there is meaningful, and that is where the flag is
    raised. Degenerate triggers are counted separately so the informational number can be
    read at all.

    Returns
    -------
    dict
        Per-channel duplication summary, written into the audit report.
    """
    subject_of = {record.recording_id: record.subject_id for record in records}
    summary: dict[str, Any] = {}

    mic_groups = duplicate_groups(
        {record.recording_id: record.mic_sorted_sha256 for record in records},
        subject_of=subject_of,
        cross_subject_only=True,
    )
    member_to_group = {
        member: fingerprint for fingerprint, members in mic_groups.items() for member in members
    }
    for record in records:
        fingerprint = member_to_group.get(record.recording_id)
        if fingerprint is None:
            continue
        siblings = [m for m in mic_groups[fingerprint] if m != record.recording_id]
        record.mic_duplicate_group = fingerprint[:8]
        record.mic_duplicate_of = ";".join(siblings)
        record.quality_flags = sorted(
            {*record.quality_flags, QualityFlag.MIC_WAVEFORM_DUPLICATED.value}
        )

    summary["mic"] = {
        "n_distinct_waveforms": len({record.mic_sorted_sha256 for record in records}),
        "n_cross_subject_groups": len(mic_groups),
        "n_recordings_affected": sum(1 for record in records if record.mic_duplicate_group),
        "groups": {fingerprint[:8]: members for fingerprint, members in sorted(mic_groups.items())},
    }

    for index in range(len(records[0].emg_sorted_sha256) if records else 0):
        emg_groups = duplicate_groups(
            {record.recording_id: record.emg_sorted_sha256[index] for record in records},
            subject_of=subject_of,
            cross_subject_only=True,
        )
        summary[f"emg{index + 1}"] = {
            "n_distinct_waveforms": len({r.emg_sorted_sha256[index] for r in records}),
            "n_cross_subject_groups": len(emg_groups),
            "groups": {f[:8]: m for f, m in sorted(emg_groups.items())},
        }
        if emg_groups:
            logger.error(
                "EMG channel %d shares waveforms across participants: %s. This is not the "
                "microphone defect -- it would invalidate every result in the project. "
                "Investigate before running anything.",
                index + 1,
                sorted(emg_groups),
                extra={"channel": index + 1, "groups": sorted(emg_groups)},
            )

    trigger_groups = duplicate_groups(
        {record.recording_id: record.trigger_sorted_sha256 for record in records},
        subject_of=subject_of,
        cross_subject_only=True,
    )
    # A trigger that is constant -- all off in a rest recording, all on in a recording that
    # was active end to end -- necessarily matches every other constant trigger. That is the
    # protocol working, not a duplication.
    degenerate = {
        fingerprint
        for fingerprint, members in trigger_groups.items()
        if all(
            next(r for r in records if r.recording_id == m).n_trigger_runs in (0, 1)
            and next(r for r in records if r.recording_id == m).trigger_active_fraction
            in (0.0, 1.0)
            for m in members
        )
    }
    summary["trigger"] = {
        "informational": True,
        "note": (
            "A binary channel's sorted-sample fingerprint collides on duty cycle alone, so "
            "a shared trigger fingerprint is not evidence of copied data. Reported for "
            "completeness; no flag is raised from it."
        ),
        "n_distinct_waveforms": len({record.trigger_sorted_sha256 for record in records}),
        "n_cross_subject_groups": len(trigger_groups),
        "n_degenerate_groups_allowed": len(degenerate),
        "groups": {
            fingerprint[:8]: members
            for fingerprint, members in sorted(trigger_groups.items())
            if fingerprint not in degenerate
        },
    }

    affected = summary["mic"]["n_recordings_affected"]
    if affected:
        logger.warning(
            "microphone waveform shared across participants in %d of %d recordings "
            "(%d distinct waveforms in total); audio-consuming runs are blocked unless "
            "the experiment declares mic_defect_acknowledged_by. See audio.md.",
            affected,
            len(records),
            summary["mic"]["n_distinct_waveforms"],
            extra={"n_affected": affected, "n_records": len(records)},
        )
    return summary


def build_manifest(
    data_root: Path | str,
    *,
    sampling_rate_hz: int = NOMINAL_SAMPLING_RATE_HZ,
    policy: ExclusionPolicy | None = None,
    probe_video: bool = True,
    hash_video: bool = False,
    filter_config: FilterChainConfig | None = None,
) -> DatasetManifest:
    """Scan ``data_root`` and build a validated manifest of every recording.

    Parameters
    ----------
    data_root
        Directory holding ``Subject_<n>/`` folders (and optionally ``More Data/``).
    sampling_rate_hz
        Configured nominal rate. Recordings whose metadata disagrees are flagged and
        excluded rather than resampled.
    policy
        Exclusion policy; defaults to :class:`ExclusionPolicy`.
    probe_video
        Read frame count/rate/size from each ``.avi``. Costs a few seconds for 100 files.
    hash_video
        SHA-256 every ``.avi``. Off by default -- the videos total ~1.5 GB.
    filter_config
        Chain used only to price the microphone columns: how much of the channel's variance
        an analysis chain retains, and how that compares to the quantisation floor. Defaults
        to the production chain. It does **not** affect any other manifest column -- the
        mains-contamination and range figures are measured on the raw signal by design --
        and the chain used is recorded per record in ``mic_chain_description``.

    Returns
    -------
    DatasetManifest
        With ``manifest_hash`` derived from the recording checksums, conditions,
        exclusions and policy version.
    """
    root = Path(data_root).resolve()
    active_policy = policy or ExclusionPolicy()
    active_filters = filter_config or FilterChainConfig()
    discovered = discover_recordings(root)
    logger.info(
        "discovered %d recordings under %s",
        len(discovered),
        root.name,
        extra={"n_recordings": len(discovered)},
    )

    records = list(
        progress.track(
            _iter_records(
                discovered,
                root,
                sampling_rate_hz,
                active_policy,
                probe_video=probe_video,
                hash_video=hash_video,
                filter_config=active_filters,
            ),
            "reading recordings",
            total=len(discovered),
            unit="file",
        )
    )
    # Second pass. Must run before the hash: a flag raised here belongs to the manifest's
    # identity, so a dataset that gains a duplicated waveform gets a different hash.
    duplication = flag_shared_waveforms(records)
    manifest_hash = hash_mapping(_summarise_for_hash(records))
    logger.info(
        "manifest built: %d records, %d included, hash %s",
        len(records),
        sum(1 for record in records if not record.excluded),
        manifest_hash,
        extra={"manifest_hash": manifest_hash},
    )
    return DatasetManifest(
        records=records,
        data_root=root,
        sampling_rate_hz=sampling_rate_hz,
        policy=active_policy,
        manifest_hash=manifest_hash,
        duplication=duplication,
    )


def load_manifest(
    path: Path | str,
    *,
    data_root: Path | str,
    policy: ExclusionPolicy | None = None,
) -> DatasetManifest:
    """Rehydrate a manifest previously written as Parquet or CSV.

    Raises
    ------
    ValueError
        If the stored manifest schema version differs from this code's version, which
        would silently mix incompatible artifacts.
    """
    source = Path(path)
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    versions = set(frame["manifest_schema_version"].astype(str))
    if versions != {MANIFEST_SCHEMA_VERSION}:
        raise ValueError(
            f"manifest at {source} was written with schema version(s) {sorted(versions)}, "
            f"but this code expects {MANIFEST_SCHEMA_VERSION}; rebuild the manifest"
        )

    records: list[RecordingRecord] = []
    for row in frame.to_dict(orient="records"):
        payload = dict(row)
        payload["columns"] = tuple(str(payload["columns"]).split(","))
        payload["trigger_values"] = tuple(
            float(value) for value in str(payload["trigger_values"]).split(",") if value
        )
        payload["quality_flags"] = [
            value for value in str(payload["quality_flags"]).split(",") if value
        ]
        payload["conflict_rules_applied"] = [
            value for value in str(payload["conflict_rules_applied"]).split(",") if value
        ]
        payload["emg_min"] = [float(v) for v in str(payload["emg_min"]).split(",") if v]
        payload["emg_max"] = [float(v) for v in str(payload["emg_max"]).split(",") if v]
        payload["emg_sorted_sha256"] = [
            v for v in str(payload["emg_sorted_sha256"]).split(",") if v
        ]
        for key in ("mic_duplicate_group", "mic_duplicate_of"):
            if pd.isna(payload.get(key)):
                payload[key] = ""
        payload["mains_harmonic_power_fraction_per_channel"] = [
            float(v)
            for v in str(payload["mains_harmonic_power_fraction_per_channel"]).split(",")
            if v
        ]
        payload["trigger_run_boundaries"] = [
            {"start_sample": int(span.split("-")[0]), "end_sample": int(span.split("-")[1])}
            for span in str(payload["trigger_run_boundaries"]).split(";")
            if span and "-" in span
        ]
        payload["excluded"] = bool(payload["excluded"])
        for key in ("avi_relpath", "metadata_relpath", "npy_relpath", "npy_disagreement"):
            if pd.isna(payload.get(key)):
                payload[key] = None
        records.append(RecordingRecord(**{str(k): v for k, v in payload.items()}))

    return DatasetManifest(
        records=records,
        data_root=Path(data_root).resolve(),
        sampling_rate_hz=int(frame["sampling_rate_hz"].iloc[0]),
        policy=policy or ExclusionPolicy(),
        manifest_hash=hash_mapping(_summarise_for_hash(records)),
    )
