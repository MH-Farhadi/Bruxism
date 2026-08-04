"""Segmentation policy, window arithmetic and the task taxonomy."""

from __future__ import annotations

import pytest

from bruxism.data.labels import TASK_DEFINITIONS, TaskFamily, family_of_token, get_task
from bruxism.data.manifest import RecordingRecord
from bruxism.data.segments import (
    SegmentationConfig,
    SegmentationPolicy,
    build_window_index,
    segments_for_recording,
)

# ------------------------------------------------------------------ taxonomy ---


@pytest.mark.parametrize(
    ("token", "family"),
    [
        ("rest", TaskFamily.REST),
        ("open_close", TaskFamily.MOVEMENT),
        ("deviation_left_right", TaskFamily.MOVEMENT),
        ("protrusion_retrusion", TaskFamily.MOVEMENT),
        ("bite_left", TaskFamily.CLENCH),
        ("bite_right", TaskFamily.CLENCH),
        ("molar_clench", TaskFamily.CLENCH),
        ("incisor_clench", TaskFamily.CLENCH),
        ("natural_bruxing", TaskFamily.INSTRUCTED_GRINDING),
        ("cheese", TaskFamily.CHEWING),
        ("carrots", TaskFamily.CHEWING),
        ("gum", TaskFamily.CHEWING),
    ],
)
def test_task_family_mapping(token, family):
    assert family_of_token(token) is family


def test_natural_bruxing_is_reported_as_instructed_grinding():
    from bruxism.data.labels import RAW_TOKEN_TO_CONDITION

    assert RAW_TOKEN_TO_CONDITION["natural_bruxing"] == "instructed_grinding"


def test_unknown_token_raises_rather_than_being_dropped():
    with pytest.raises(KeyError, match="unknown condition token"):
        family_of_token("humming")


@pytest.mark.parametrize("task_id", sorted(TASK_DEFINITIONS))
def test_every_task_is_internally_consistent(task_id):
    task = get_task(task_id)
    assert len(task.class_names) == len(set(task.class_names))
    assert set(task.family_to_class.values()) == set(task.class_names)
    for family in task.family_to_class:
        label = task.label_for_family(family)
        assert label is not None and 0 <= label < task.num_classes


def test_binary_collapse_is_correct():
    task = get_task("binary_tooth_contact")
    positive = task.class_to_index["tooth_contact"]
    assert task.label_for_family(TaskFamily.CLENCH) == positive
    assert task.label_for_family(TaskFamily.INSTRUCTED_GRINDING) == positive
    for family in (TaskFamily.REST, TaskFamily.MOVEMENT, TaskFamily.CHEWING):
        assert task.label_for_family(family) != positive


def test_ternary_collapse_is_correct():
    task = get_task("ternary")
    assert task.label_for_family(TaskFamily.CLENCH) == task.class_to_index["tooth_contact"]
    assert (
        task.label_for_family(TaskFamily.INSTRUCTED_GRINDING)
        == task.class_to_index["tooth_contact"]
    )
    assert task.label_for_family(TaskFamily.CHEWING) == task.class_to_index["chewing"]
    assert (
        task.label_for_family(TaskFamily.REST)
        == task.label_for_family(TaskFamily.MOVEMENT)
        == task.class_to_index["rest_or_movement"]
    )


def test_no_chewing_task_excludes_rather_than_merges_chewing():
    task = get_task("no_chewing_four_class")
    assert task.label_for_family(TaskFamily.CHEWING) is None


def test_legacy_task_is_not_a_primary_endpoint():
    assert get_task("legacy_active_four_class").primary is False
    assert get_task("five_class").primary is True


# ------------------------------------------------------------- segmentation ---


def _record(n_samples: int, runs: list[tuple[int, int]], token: str = "molar_clench"):
    return RecordingRecord(
        recording_id="S01_x_20250804T100000",
        subject_id="S01",
        condition_token=token,
        condition=token,
        task_family="clench",
        repetition_token="t",
        csv_relpath="a.csv",
        avi_relpath=None,
        metadata_relpath=None,
        n_samples=n_samples,
        sampling_rate_hz=1200,
        metadata_sampling_rate_hz=1200,
        duration_seconds=n_samples / 1200,
        columns=(),
        trigger_values=(0.0, 1.0),
        trigger_active_samples=sum(e - s for s, e in runs),
        trigger_active_fraction=0.5,
        n_trigger_runs=len(runs),
        n_trigger_transitions=2 * len(runs),
        trigger_run_boundaries=[{"start_sample": s, "end_sample": e} for s, e in runs],
        emg_min=[0.0],
        emg_max=[1.0],
        mic_min=0.0,
        mic_max=1.0,
        csv_sha256="x",
        avi_sha256=None,
        metadata_sha256=None,
        metadata_condition_key=None,
        metadata_status="COMPLETED",
        metadata_samples_saved=n_samples,
        metadata_target_duration_seconds=n_samples / 1200,
        npy_relpath=None,
        npy_claimed=False,
        npy_agrees_with_csv=None,
        npy_disagreement=None,
        video_frame_count=None,
        video_fps=None,
        video_duration_seconds=None,
        video_width=None,
        video_height=None,
        video_codec=None,
        video_readable=False,
        startup_transient_seconds=0.0,
        startup_transient_peak_ratio=0.0,
        mains_harmonic_power_fraction=0.0,
        mains_harmonic_power_fraction_per_channel=[0.0, 0.0, 0.0, 0.0],
        mains_harmonic_breakdown="",
        quality_flags=[],
        excluded=False,
        exclusion_reason="",
        quality_policy_version="test",
    )


def test_guard_shrinks_runs_symmetrically():
    # One 5 s run inside a 10 s recording, 0.5 s guard on each side -> 4 s usable.
    record = _record(12000, [(2400, 8400)])
    config = SegmentationConfig(guard_seconds=0.5, startup_guard_seconds=0.0)
    segments = segments_for_recording(record, config, sampling_rate=1200)
    assert len(segments) == 1
    assert segments[0].start_sample == 2400 + 600
    assert segments[0].end_sample == 8400 - 600


def test_run_shorter_than_window_plus_guards_yields_nothing():
    # 1.5 s run, 1 s window, 0.5 s guard each side -> 0.5 s usable, no window fits.
    record = _record(12000, [(2400, 4200)])
    config = SegmentationConfig(guard_seconds=0.5, startup_guard_seconds=0.0)
    segments = segments_for_recording(record, config, sampling_rate=1200)
    assert all(s.n_samples < 1200 for s in segments)


def test_window_arithmetic_is_exact():
    # A 4 s segment with a 1 s window and 0.5 s stride yields floor(3/0.5)+1 = 7 windows.
    record = _record(12000, [(1200, 7200)])  # 5 s run -> 4 s after 0.5 s guards
    index = build_window_index(
        _manifest_of(record),
        SegmentationConfig(guard_seconds=0.5, startup_guard_seconds=0.0),
    )
    assert len(index.windows) == 7
    assert all(w.end_sample - w.start_sample == 1200 for w in index.windows)


def test_windows_never_escape_their_segment():
    record = _record(12000, [(2400, 8400)])
    index = build_window_index(
        _manifest_of(record),
        SegmentationConfig(guard_seconds=0.5, startup_guard_seconds=0.0),
    )
    for window in index.windows:
        assert window.start_sample >= 3000
        assert window.end_sample <= 7800


def test_sample_ids_are_unique_and_stable():
    record = _record(12000, [(1200, 7200)])
    config = SegmentationConfig(guard_seconds=0.5, startup_guard_seconds=0.0)
    first = build_window_index(_manifest_of(record), config)
    second = build_window_index(_manifest_of(record), config)
    ids = [w.sample_id for w in first.windows]
    assert len(ids) == len(set(ids))
    assert ids == [w.sample_id for w in second.windows]


def test_short_recording_windowing(synthetic_manifest):
    short = [r for r in synthetic_manifest.records if "short_recording" in r.quality_flags]
    assert short
    index = build_window_index(
        synthetic_manifest, SegmentationConfig(guard_seconds=0.25, startup_guard_seconds=0.5)
    )
    for window in index.windows:
        record = synthetic_manifest.by_id(window.recording_id)
        assert window.end_sample <= record.n_samples


def test_rest_comes_only_from_dedicated_recordings_by_default(synthetic_window_index):
    rest = [w for w in synthetic_window_index.windows if w.task_family == "rest"]
    assert rest
    assert {w.segment_source for w in rest} == {"dedicated_rest"}


def test_trigger_off_as_rest_requires_approval():
    with pytest.raises(ValueError, match="requires trigger_off_rest_approved_by"):
        SegmentationConfig(allow_trigger_off_as_rest=True)
    # With an approval recorded it is permitted.
    config = SegmentationConfig(
        allow_trigger_off_as_rest=True, trigger_off_rest_approved_by="PI, 2026-07-27"
    )
    assert config.allow_trigger_off_as_rest


def test_legacy_policy_is_marked_unsafe_for_inference(synthetic_manifest):
    legacy = build_window_index(
        synthetic_manifest,
        SegmentationConfig(
            policy=SegmentationPolicy.WHOLE_RECORDING_LEGACY, startup_guard_seconds=0.0
        ),
    )
    assert legacy.safe_for_inference is False
    assert all(w.safe_for_inference is False for w in legacy.windows)
    assert all(w.segment_source == "legacy_whole_recording" for w in legacy.windows)


def test_trigger_constrained_policy_is_safe(synthetic_window_index):
    assert synthetic_window_index.safe_for_inference is True


def test_stride_larger_than_window_is_rejected():
    with pytest.raises(ValueError, match="exceeds window_seconds"):
        SegmentationConfig(window_seconds=1.0, stride_seconds=2.0)


def _manifest_of(record):
    from pathlib import Path

    from bruxism.data.manifest import DatasetManifest
    from bruxism.data.quality import ExclusionPolicy

    return DatasetManifest(
        records=[record],
        data_root=Path(),
        sampling_rate_hz=1200,
        policy=ExclusionPolicy(),
        manifest_hash="test",
    )
