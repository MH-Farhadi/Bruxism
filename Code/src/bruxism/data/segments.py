"""Segmentation policies: turning continuous recordings into labelled windows.

Two policies exist and they are not interchangeable.

``trigger_constrained``
    The approved scientific policy. A window is emitted only when it lies wholly inside a
    single homogeneous approved segment, at least ``guard_seconds`` away from every
    trigger transition and from the recording's startup/shutdown guards. The investigator
    has confirmed that ``Trigger == 1`` marks intervals during which the participant was
    actively performing the named task (recorded 2026-07-27, see
    ``docs/experiment_protocol.md``).

``whole_recording_legacy``
    Diagnostic only. Labels every window of an active recording by its filename, ignoring
    the trigger entirely -- the behaviour of the original prototype. It exists to
    reproduce and explain historical window counts and is marked unsafe for inference; any
    :class:`WindowIndex` built with it carries ``safe_for_inference = False``.

Windows never overlap across segments, and a window is emitted at most once, so the same
samples can never appear as two examples or land in two folds.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from bruxism.data.labels import TaskFamily, family_of_token
from bruxism.data.manifest import DatasetManifest, RecordingRecord, TriggerRun
from bruxism.utils.io import hash_mapping
from bruxism.utils.logging import get_logger

__all__ = [
    "SegmentationConfig",
    "SegmentationPolicy",
    "SegmentSource",
    "WindowIndex",
    "WindowRecord",
    "build_window_index",
    "segments_for_recording",
]

logger = get_logger(__name__)


class SegmentationPolicy(StrEnum):
    """Named segmentation policies."""

    TRIGGER_CONSTRAINED = "trigger_constrained"
    WHOLE_RECORDING_LEGACY = "whole_recording_legacy"


class SegmentSource(StrEnum):
    """Where a window's label came from, recorded on every example."""

    #: Inside a trigger-high run of an active-task recording.
    TRIGGER_ACTIVE = "trigger_active"
    #: Anywhere inside a dedicated rest recording (whose trigger is flat zero).
    DEDICATED_REST = "dedicated_rest"
    #: Trigger-low interval of an active recording. Requires explicit approval.
    TRIGGER_OFF_AS_REST = "trigger_off_as_rest"
    #: Whole-recording filename labelling; diagnostic only.
    LEGACY_WHOLE_RECORDING = "legacy_whole_recording"


@dataclass(frozen=True)
class SegmentationConfig:
    """Parameters of a segmentation policy.

    Attributes
    ----------
    policy
        Which policy to apply.
    window_seconds, stride_seconds
        Observation window and decision interval. The protocol default is 1.0 s / 0.5 s.
    guard_seconds
        Interval excluded on *each* side of every trigger transition, so a window never
        straddles an onset or offset. Approved default 0.5 s.
    startup_guard_seconds, shutdown_guard_seconds
        Declared quality rule replacing the prototype's unexplained 3 s skip. Samples
        within these guards of the recording boundaries are never used, because amplifier
        settling and acquisition teardown are visible there.
    allow_trigger_off_as_rest
        Must stay ``False`` unless an investigator has confirmed that trigger-low
        intervals in active recordings are quiet rest rather than transitions or
        instruction periods. Setting it ``True`` without ``trigger_off_rest_approved_by``
        raises.
    trigger_off_rest_approved_by
        Name and date of the approval, recorded in the run bundle.
    min_segment_seconds
        Segments shorter than this cannot host a full window and are dropped with a reason.
    """

    policy: SegmentationPolicy = SegmentationPolicy.TRIGGER_CONSTRAINED
    window_seconds: float = 1.0
    stride_seconds: float = 0.5
    guard_seconds: float = 0.5
    startup_guard_seconds: float = 1.0
    shutdown_guard_seconds: float = 0.0
    allow_trigger_off_as_rest: bool = False
    trigger_off_rest_approved_by: str | None = None
    min_segment_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.window_seconds <= 0 or self.stride_seconds <= 0:
            raise ValueError("window_seconds and stride_seconds must be positive")
        if self.stride_seconds > self.window_seconds:
            raise ValueError(
                f"stride_seconds ({self.stride_seconds}) exceeds window_seconds "
                f"({self.window_seconds}); that would skip samples entirely"
            )
        if self.guard_seconds < 0 or self.startup_guard_seconds < 0:
            raise ValueError("guard intervals must be non-negative")
        if self.allow_trigger_off_as_rest and not self.trigger_off_rest_approved_by:
            raise ValueError(
                "allow_trigger_off_as_rest=True requires trigger_off_rest_approved_by; "
                "trigger-off intervals may be transitions or instruction periods, and "
                "treating them as rest without approval fabricates a class"
            )

    def window_samples(self, sampling_rate: int) -> int:
        return int(round(self.window_seconds * sampling_rate))

    def stride_samples(self, sampling_rate: int) -> int:
        return int(round(self.stride_seconds * sampling_rate))

    def guard_samples(self, sampling_rate: int) -> int:
        return int(round(self.guard_seconds * sampling_rate))

    @property
    def safe_for_inference(self) -> bool:
        """Whether results from this policy may be reported as scientific findings."""
        return self.policy is SegmentationPolicy.TRIGGER_CONSTRAINED

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy"] = str(self.policy)
        payload["safe_for_inference"] = self.safe_for_inference
        return payload


@dataclass(frozen=True)
class _Segment:
    """A homogeneous interval of one recording that may host windows."""

    start_sample: int
    end_sample: int
    family: TaskFamily
    source: SegmentSource
    trigger_run_index: int | None

    @property
    def n_samples(self) -> int:
        return self.end_sample - self.start_sample


def _clip(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def segments_for_recording(
    record: RecordingRecord,
    config: SegmentationConfig,
    *,
    sampling_rate: int,
) -> list[_Segment]:
    """Compute the approved homogeneous segments of one recording.

    Under ``trigger_constrained`` these are the trigger-high runs of an active recording
    (shrunk by ``guard_seconds`` on each side) and the whole span of a dedicated rest
    recording (shrunk by the startup/shutdown guards). Under ``whole_recording_legacy``
    it is one segment covering the whole recording.
    """
    family = family_of_token(record.condition_token)
    n = record.n_samples
    startup = int(round(config.startup_guard_seconds * sampling_rate))
    shutdown = int(round(config.shutdown_guard_seconds * sampling_rate))
    usable_start, usable_end = startup, n - shutdown
    if usable_end <= usable_start:
        return []

    if config.policy is SegmentationPolicy.WHOLE_RECORDING_LEGACY:
        # Deliberately ignores the trigger and the guards, reproducing the prototype.
        return [
            _Segment(
                start_sample=0,
                end_sample=n,
                family=family,
                source=SegmentSource.LEGACY_WHOLE_RECORDING,
                trigger_run_index=None,
            )
        ]

    guard = config.guard_samples(sampling_rate)
    segments: list[_Segment] = []

    if family is TaskFamily.REST:
        # Dedicated rest recordings: the trigger is flat zero throughout, so the whole
        # usable span is one homogeneous rest segment.
        segments.append(
            _Segment(
                start_sample=usable_start,
                end_sample=usable_end,
                family=TaskFamily.REST,
                source=SegmentSource.DEDICATED_REST,
                trigger_run_index=None,
            )
        )
        return segments

    runs = [
        TriggerRun(index=i, start_sample=b["start_sample"], end_sample=b["end_sample"])
        for i, b in enumerate(record.trigger_run_boundaries)
    ]
    for run in runs:
        start = _clip(run.start_sample + guard, usable_start, usable_end)
        end = _clip(run.end_sample - guard, usable_start, usable_end)
        if end > start:
            segments.append(
                _Segment(
                    start_sample=start,
                    end_sample=end,
                    family=family,
                    source=SegmentSource.TRIGGER_ACTIVE,
                    trigger_run_index=run.index,
                )
            )

    if config.allow_trigger_off_as_rest:
        # Complement of the trigger-high runs, guarded on both sides. Only reachable when
        # an investigator has approved the semantics.
        cursor = usable_start
        boundaries = [*[(run.start_sample, run.end_sample) for run in runs], (n, n)]
        for run_start, run_end in boundaries:
            gap_start = _clip(cursor + guard, usable_start, usable_end)
            gap_end = _clip(run_start - guard, usable_start, usable_end)
            if gap_end > gap_start:
                segments.append(
                    _Segment(
                        start_sample=gap_start,
                        end_sample=gap_end,
                        family=TaskFamily.REST,
                        source=SegmentSource.TRIGGER_OFF_AS_REST,
                        trigger_run_index=None,
                    )
                )
            cursor = run_end

    segments.sort(key=lambda segment: segment.start_sample)
    return segments


@dataclass(frozen=True)
class WindowRecord:
    """One labelled example, fully traceable back to raw samples."""

    sample_id: str
    subject_id: str
    recording_id: str
    condition: str
    condition_token: str
    task_family: str
    repetition_token: str
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    trigger_run_index: int | None
    segment_source: str
    segmentation_policy: str
    safe_for_inference: bool

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WindowIndex:
    """The complete set of examples produced by one segmentation configuration.

    This is an *index*: it stores sample boundaries, never signal data. Loading is lazy
    and happens in :mod:`bruxism.data.dataset`.
    """

    windows: list[WindowRecord]
    config: SegmentationConfig
    sampling_rate_hz: int
    manifest_hash: str
    dropped: list[dict[str, Any]] = field(default_factory=list)

    @property
    def safe_for_inference(self) -> bool:
        return self.config.safe_for_inference

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([window.to_row() for window in self.windows])

    @property
    def index_hash(self) -> str:
        """Identity of this window set: manifest + policy + the window boundaries."""
        return hash_mapping(
            {
                "manifest_hash": self.manifest_hash,
                "config": self.config.to_dict(),
                "sampling_rate_hz": self.sampling_rate_hz,
                "sample_ids": sorted(window.sample_id for window in self.windows),
            }
        )

    @property
    def subject_ids(self) -> list[str]:
        return sorted({window.subject_id for window in self.windows})

    def counts_by(self, *columns: str) -> pd.DataFrame:
        """Example counts grouped by any manifest columns, for the flow diagram."""
        frame = self.frame
        if frame.empty:
            return pd.DataFrame(columns=[*columns, "n_windows"])
        return (
            frame.groupby(list(columns), observed=True)
            .size()
            .reset_index(name="n_windows")
            .sort_values(list(columns), ignore_index=True)
        )

    def assert_no_overlap_between(self, group_a: Sequence[str], group_b: Sequence[str]) -> None:
        """Raise if any sample id appears in both groups.

        Used as a hard leakage guard around every train/validation/test split.
        """
        overlap = set(group_a) & set(group_b)
        if overlap:
            example = sorted(overlap)[:5]
            raise AssertionError(
                f"{len(overlap)} sample ids appear in both groups (e.g. {example}); "
                f"this is a leakage bug, not a configuration choice"
            )


def _iter_windows(
    record: RecordingRecord,
    segments: Sequence[_Segment],
    config: SegmentationConfig,
    sampling_rate: int,
) -> Iterator[WindowRecord]:
    window = config.window_samples(sampling_rate)
    stride = config.stride_samples(sampling_rate)
    for segment in segments:
        if segment.n_samples < window:
            continue
        n_windows = (segment.n_samples - window) // stride + 1
        for i in range(n_windows):
            start = segment.start_sample + i * stride
            end = start + window
            # Guaranteed by the arithmetic above, but asserted because a violation would
            # silently mix two conditions into one example.
            if end > segment.end_sample:
                raise AssertionError(
                    f"window [{start}, {end}) escapes segment "
                    f"[{segment.start_sample}, {segment.end_sample}) in {record.recording_id}"
                )
            yield WindowRecord(
                sample_id=f"{record.recording_id}#{start:07d}-{end:07d}",
                subject_id=record.subject_id,
                recording_id=record.recording_id,
                condition=record.condition,
                condition_token=record.condition_token,
                task_family=str(segment.family),
                repetition_token=record.repetition_token,
                start_sample=start,
                end_sample=end,
                start_seconds=start / sampling_rate,
                end_seconds=end / sampling_rate,
                trigger_run_index=segment.trigger_run_index,
                segment_source=str(segment.source),
                segmentation_policy=str(config.policy),
                safe_for_inference=config.safe_for_inference,
            )


def build_window_index(
    manifest: DatasetManifest,
    config: SegmentationConfig | None = None,
    *,
    include_excluded: bool = False,
) -> WindowIndex:
    """Apply a segmentation policy to a manifest and produce the example index.

    Parameters
    ----------
    manifest
        A validated dataset manifest.
    config
        Segmentation configuration; defaults to the approved trigger-constrained policy.
    include_excluded
        When true, recordings excluded by the quality policy still contribute windows.
        Only for diagnostics -- never for a reported result.

    Returns
    -------
    WindowIndex
        With one entry per emitted window and a ``dropped`` ledger explaining every
        recording or segment that produced none.
    """
    active = config or SegmentationConfig()
    rate = manifest.sampling_rate_hz
    window_samples = active.window_samples(rate)

    windows: list[WindowRecord] = []
    dropped: list[dict[str, Any]] = []

    records = manifest.records if include_excluded else manifest.included
    for record in sorted(records, key=lambda item: item.recording_id):
        if record.excluded and not include_excluded:
            dropped.append(
                {
                    "recording_id": record.recording_id,
                    "reason": f"recording excluded: {record.exclusion_reason}",
                    "n_windows": 0,
                }
            )
            continue
        segments = segments_for_recording(record, active, sampling_rate=rate)
        if not segments:
            dropped.append(
                {
                    "recording_id": record.recording_id,
                    "reason": "no approved segment after guards",
                    "n_windows": 0,
                }
            )
            continue
        too_short = [s for s in segments if s.n_samples < window_samples]
        for segment in too_short:
            dropped.append(
                {
                    "recording_id": record.recording_id,
                    "reason": (
                        f"segment [{segment.start_sample},{segment.end_sample}) shorter "
                        f"than one {active.window_seconds:g}s window after guards"
                    ),
                    "n_windows": 0,
                }
            )
        windows.extend(_iter_windows(record, segments, active, rate))

    seen: set[str] = set()
    for window in windows:
        if window.sample_id in seen:
            raise AssertionError(
                f"duplicate sample_id {window.sample_id!r}; window identity must be unique"
            )
        seen.add(window.sample_id)

    logger.info(
        "segmentation %s produced %d windows from %d recordings (%d drop notes)",
        active.policy,
        len(windows),
        len({window.recording_id for window in windows}),
        len(dropped),
        extra={"policy": str(active.policy), "n_windows": len(windows)},
    )
    if not active.safe_for_inference:
        logger.warning(
            "segmentation policy %s is DIAGNOSTIC ONLY and unsafe for inference",
            active.policy,
        )
    return WindowIndex(
        windows=windows,
        config=active,
        sampling_rate_hz=rate,
        manifest_hash=manifest.manifest_hash,
        dropped=dropped,
    )


def window_bounds_array(index: WindowIndex) -> np.ndarray:
    """``(n_windows, 2)`` int64 array of ``[start_sample, end_sample)`` pairs."""
    if not index.windows:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(
        [[window.start_sample, window.end_sample] for window in index.windows], dtype=np.int64
    )
