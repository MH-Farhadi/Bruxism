"""Data-quality flags, conflict resolutions and exclusion policy.

Every anomaly found in the raw dataset is represented as a named flag with an explicit,
versioned resolution rule. Nothing is silently repaired: a recording whose metadata
contradicts its filename keeps both values in the manifest plus the flag and the rule
that was applied.

Bumping :data:`QUALITY_POLICY_VERSION` invalidates every previously built manifest, which
is intentional -- artifacts record the policy version they were built under.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

__all__ = [
    "CONFLICT_RESOLUTIONS",
    "QUALITY_POLICY_VERSION",
    "ConflictResolution",
    "ExclusionPolicy",
    "QualityFlag",
    "describe_flag",
]

#: Increment whenever a flag definition, threshold or resolution rule changes.
#: 2026-08-03.1 adds MAINS_CONTAMINATION.
QUALITY_POLICY_VERSION: Final[str] = "2026-08-03.1"


class QualityFlag(StrEnum):
    """Named data-quality conditions detected during manifest construction."""

    #: Recording has fewer samples than the 60 s target duration.
    SHORT_RECORDING = "short_recording"
    #: File lives outside the participant's primary directory (``Data/More Data/...``).
    SECONDARY_LOCATION = "secondary_location"
    #: ``condition_key`` in the metadata disagrees with the filename token.
    METADATA_CONDITION_CONFLICT = "metadata_condition_conflict"
    #: Metadata claims an ``.npy`` companion that does not exist on disk.
    MISSING_NPY_COMPANION = "missing_npy_companion"
    #: An ``.npy`` companion exists but does not reproduce the CSV contents.
    STALE_NPY_COMPANION = "stale_npy_companion"
    #: The CSV/AVI/metadata triple is incomplete.
    INCOMPLETE_TRIPLE = "incomplete_triple"
    #: Metadata sampling rate differs from the configured nominal rate.
    SAMPLING_RATE_MISMATCH = "sampling_rate_mismatch"
    #: ``samples_saved`` in the metadata differs from the CSV row count.
    SAMPLE_COUNT_MISMATCH = "sample_count_mismatch"
    #: Video duration differs from the CSV duration by more than the tolerance.
    VIDEO_DURATION_MISMATCH = "video_duration_mismatch"
    #: Video file could not be probed (missing codec support or unreadable file).
    VIDEO_UNREADABLE = "video_unreadable"
    #: Large amplitude excursion in the first seconds, consistent with amplifier settling.
    STARTUP_TRANSIENT = "startup_transient"
    #: Trigger never activates in a recording whose condition is an active task.
    NO_TRIGGER_ACTIVITY = "no_trigger_activity"
    #: Trigger activates in a dedicated rest recording (should never happen).
    UNEXPECTED_TRIGGER_IN_REST = "unexpected_trigger_in_rest"
    #: Metadata ``status`` is not ``COMPLETED``.
    NOT_COMPLETED = "not_completed"
    #: Most of the raw in-band EMG power sits at multiples of the mains frequency.
    MAINS_CONTAMINATION = "mains_contamination"


_FLAG_DESCRIPTIONS: Final[dict[QualityFlag, str]] = {
    QualityFlag.SHORT_RECORDING: (
        "Materially fewer samples than the target duration declared in the recording's own "
        "metadata. Retained: windowing handles variable length, and the shortfall is "
        "reported per recording."
    ),
    QualityFlag.SECONDARY_LOCATION: (
        "File is stored under 'Data/More Data/Data/<Subject>/' rather than the "
        "participant's primary directory. Retained; the physical path is recorded verbatim "
        "and the file is never moved."
    ),
    QualityFlag.METADATA_CONDITION_CONFLICT: (
        "The metadata 'condition_key' disagrees with the filename condition token. "
        "Resolved by the named rule in CONFLICT_RESOLUTIONS; both values are kept."
    ),
    QualityFlag.MISSING_NPY_COMPANION: (
        "Metadata names an .npy companion that does not exist. Harmless: the pipeline "
        "reads CSV only and regenerates its own caches."
    ),
    QualityFlag.STALE_NPY_COMPANION: (
        "An .npy companion exists but does not reproduce the CSV. Treated as a stale "
        "acquisition-time cache and never used as an input."
    ),
    QualityFlag.INCOMPLETE_TRIPLE: (
        "One or more of the CSV / AVI / metadata triple is absent. Recording is retained "
        "if the CSV exists; missing companions are recorded as nulls."
    ),
    QualityFlag.SAMPLING_RATE_MISMATCH: (
        "Metadata sampling rate differs from the configured nominal rate. Excluded: every "
        "filter cutoff and window length in this pipeline assumes a single rate."
    ),
    QualityFlag.SAMPLE_COUNT_MISMATCH: (
        "Metadata 'samples_saved' differs from the CSV row count. The CSV row count wins "
        "(it is the actual data); the discrepancy is reported."
    ),
    QualityFlag.VIDEO_DURATION_MISMATCH: (
        "Video and CSV durations differ by more than the tolerance. Informational only -- "
        "video is not an input to any model in this pipeline."
    ),
    QualityFlag.VIDEO_UNREADABLE: ("Video metadata could not be probed. Informational only."),
    QualityFlag.STARTUP_TRANSIENT: (
        "Amplitude in the opening seconds far exceeds the recording's robust scale, "
        "consistent with amplifier settling. Handled by the declared startup-guard rule in "
        "the segmentation policy, not by a blanket sample skip."
    ),
    QualityFlag.NO_TRIGGER_ACTIVITY: (
        "An active-task recording whose trigger never goes high. Excluded from the "
        "trigger-constrained protocol because it yields no approved task interval."
    ),
    QualityFlag.UNEXPECTED_TRIGGER_IN_REST: (
        "A dedicated rest recording whose trigger goes high. Excluded: the rest class "
        "depends on the recording being uniformly quiet."
    ),
    QualityFlag.NOT_COMPLETED: (
        "Metadata 'status' is not COMPLETED. Excluded: the acquisition run did not finish."
    ),
    QualityFlag.MAINS_CONTAMINATION: (
        "More than 30 % of the raw 20-450 Hz EMG power sits within 3 Hz of a multiple of "
        "the mains frequency: the channel is measuring electrode impedance and cable "
        "routing more than muscle. RETAINED, NOT EXCLUDED. It is a flag for a human, not "
        "an exclusion rule -- dropping the affected participant automatically would have "
        "hidden the finding that the production filter chain notched 60 Hz, which the "
        "hardware had already removed, and passed 180/300/420 Hz untouched (cause.md). "
        "The corrected chain notches every harmonic; this flag describes the raw signal "
        "and stays raised afterwards, because what it records is a property of the "
        "acquisition, not of the analysis."
    ),
}


def describe_flag(flag: QualityFlag) -> str:
    """Human-readable rationale for a flag, used verbatim in the audit report."""
    return _FLAG_DESCRIPTIONS[flag]


@dataclass(frozen=True)
class ConflictResolution:
    """A named, reviewable rule for resolving one specific data conflict."""

    rule_id: str
    applies_to: str
    conflict: str
    resolution: str
    rationale: str
    approved_by: str
    approved_on: str


CONFLICT_RESOLUTIONS: Final[dict[str, ConflictResolution]] = {
    res.rule_id: res
    for res in (
        ConflictResolution(
            rule_id="R1_filename_wins_for_condition",
            applies_to="S05 molar_clench_5_20250807_145916 (and any future identical case)",
            conflict=(
                "The metadata for S05's molar-clench recording reports "
                "'condition_key: incisor_clench', which is also the condition_key of S05's "
                "separate incisor-clench recording at 15:00:41."
            ),
            resolution=(
                "The filename condition token wins. The recording is labelled molar_clench; "
                "the conflicting metadata value is preserved in the manifest column "
                "'metadata_condition_key' and the recording carries "
                "QualityFlag.METADATA_CONDITION_CONFLICT."
            ),
            rationale=(
                "S05 has both a molar-clench and an incisor-clench recording, and both "
                "metadata files claim 'incisor_clench'. Trusting the metadata would give the "
                "participant two incisor recordings and no molar recording, contradicting the "
                "protocol described in the manuscript, which specifies both tasks. The "
                "duplicated value is consistent with a stale field carried over between "
                "consecutive acquisitions."
            ),
            approved_by="Project owner (session decision)",
            approved_on="2026-07-27",
        ),
        ConflictResolution(
            rule_id="R2_csv_is_authoritative_over_npy",
            applies_to="All recordings; three .npy files exist, all for S01",
            conflict=(
                "Every metadata file names an .npy companion, but only three exist. In those "
                "three the first five columns match EMG+Trigger while the sixth column is "
                "all zeros and does not reproduce the CSV microphone channel."
            ),
            resolution=(
                "The CSV is the sole data source. Existing .npy files are never read as "
                "input; any cache used by the pipeline is regenerated from the CSV checksum "
                "plus the resolved preprocessing configuration."
            ),
            rationale=(
                "The .npy files are incomplete acquisition-time caches that would silently "
                "zero the microphone channel."
            ),
            approved_by="sol.md implementation brief",
            approved_on="2026-07-27",
        ),
        ConflictResolution(
            rule_id="R3_csv_row_count_is_authoritative",
            applies_to="All recordings",
            conflict=(
                "Metadata reports 'samples_saved' and 'expected_samples'; these disagree with "
                "each other and occasionally with the CSV row count."
            ),
            resolution=(
                "Duration and windowing are derived from the CSV row count divided by the "
                "configured sampling rate. Metadata values are recorded for audit only."
            ),
            rationale="Only the CSV row count describes data that actually exists.",
            approved_by="sol.md implementation brief",
            approved_on="2026-07-27",
        ),
    )
}


@dataclass(frozen=True)
class ExclusionPolicy:
    """Which quality flags remove a recording from analysis.

    ``policy_version`` is stamped on every manifest and every prediction row so a metric
    can always be traced back to the exclusion rules that produced its sample set.
    """

    policy_version: str = QUALITY_POLICY_VERSION
    #: Flags that exclude a recording outright.
    excluding_flags: frozenset[QualityFlag] = frozenset(
        {
            QualityFlag.SAMPLING_RATE_MISMATCH,
            QualityFlag.NOT_COMPLETED,
            QualityFlag.NO_TRIGGER_ACTIVITY,
            QualityFlag.UNEXPECTED_TRIGGER_IN_REST,
        }
    )
    #: Minimum usable duration in seconds; shorter recordings cannot yield a window.
    min_duration_seconds: float = 2.0
    #: A recording is "short" below this fraction of the target duration its own metadata
    #: declares. Evaluated per recording, so the rule follows the protocol rather than
    #: assuming every session targeted 60 s.
    short_recording_fraction: float = 0.98
    #: Video/CSV duration difference tolerated before flagging.
    video_duration_tolerance_seconds: float = 3.0
    #: Startup-transient detector: multiple of the robust scale that counts as an excursion.
    startup_transient_sigma: float = 12.0
    #: Window at the start of a recording examined for the startup transient.
    startup_probe_seconds: float = 3.0

    def evaluate(
        self, flags: frozenset[QualityFlag] | set[QualityFlag], duration_seconds: float
    ) -> tuple[bool, str]:
        """Decide whether a recording is excluded.

        Returns
        -------
        (excluded, reason)
            ``reason`` is an empty string when the recording is retained.
        """
        triggered = sorted(flag.value for flag in set(flags) & set(self.excluding_flags))
        if triggered:
            return True, f"excluding_flags:{','.join(triggered)}"
        if duration_seconds < self.min_duration_seconds:
            return True, (
                f"duration {duration_seconds:.2f}s below minimum {self.min_duration_seconds:.2f}s"
            )
        return False, ""
