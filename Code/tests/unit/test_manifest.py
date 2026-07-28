"""Manifest discovery, checksums and quality flags."""

from __future__ import annotations

import numpy as np
import pytest

from bruxism.data.manifest import build_manifest, discover_recordings, trigger_runs
from bruxism.data.quality import QualityFlag
from bruxism.utils.io import file_sha256


def test_discovery_is_deterministic_and_complete(synthetic_root):
    first = discover_recordings(synthetic_root)
    second = discover_recordings(synthetic_root)
    assert [k.recording_id for k, _ in first] == [k.recording_id for k, _ in second]
    assert len(first) == 5 * 12


def test_discovers_files_in_the_secondary_location(synthetic_manifest):
    secondary = [r for r in synthetic_manifest.records if "More Data" in r.csv_relpath]
    assert secondary, "the secondary-location recording was not discovered"
    assert QualityFlag.SECONDARY_LOCATION.value in secondary[0].quality_flags


def test_detects_filename_metadata_condition_conflict(synthetic_manifest):
    conflicted = [
        r
        for r in synthetic_manifest.records
        if QualityFlag.METADATA_CONDITION_CONFLICT.value in r.quality_flags
    ]
    assert len(conflicted) == 1
    record = conflicted[0]
    # The FILENAME wins, and the metadata value is preserved rather than overwritten.
    assert record.condition_token == "molar_clench"
    assert record.metadata_condition_key == "incisor_clench"
    assert "R1_filename_wins_for_condition" in record.conflict_rules_applied


def test_flags_short_recording(synthetic_manifest):
    short = [
        r
        for r in synthetic_manifest.records
        if QualityFlag.SHORT_RECORDING.value in r.quality_flags
    ]
    assert short and short[0].duration_seconds < 20.0


def test_flags_startup_transient_and_measures_it(synthetic_manifest):
    flagged = [
        r
        for r in synthetic_manifest.records
        if QualityFlag.STARTUP_TRANSIENT.value in r.quality_flags
    ]
    assert flagged
    assert all(r.startup_transient_seconds > 0 for r in flagged)
    assert all(r.startup_transient_peak_ratio > 0 for r in flagged)


def test_flags_missing_npy_companion(synthetic_manifest):
    # Every synthetic metadata claims an .npy that was never written.
    assert all(
        QualityFlag.MISSING_NPY_COMPANION.value in r.quality_flags
        for r in synthetic_manifest.records
    )
    assert all(r.npy_relpath is None for r in synthetic_manifest.records)


def test_records_checksums_durations_and_triples(synthetic_manifest):
    for record in synthetic_manifest.records:
        path = synthetic_manifest.csv_path(record)
        assert record.csv_sha256 == file_sha256(path)
        assert record.duration_seconds == pytest.approx(record.n_samples / record.sampling_rate_hz)
        assert record.avi_relpath is not None
        assert record.metadata_relpath is not None


def test_manifest_hash_is_stable_and_content_sensitive(synthetic_root, tmp_path):
    a = build_manifest(synthetic_root, probe_video=False)
    b = build_manifest(synthetic_root, probe_video=False)
    assert a.manifest_hash == b.manifest_hash

    import shutil

    copied = tmp_path / "copy"
    shutil.copytree(synthetic_root, copied)
    target = next(copied.rglob("rest_1_*.csv"))
    text = target.read_text().splitlines()
    text[1] = text[1].replace(text[1].split(",")[0], "99999.0", 1)
    target.write_text("\n".join(text) + "\n")
    assert build_manifest(copied, probe_video=False).manifest_hash != a.manifest_hash


def test_rest_recordings_have_a_flat_trigger(synthetic_manifest):
    for record in synthetic_manifest.records:
        if record.condition == "rest":
            assert record.n_trigger_runs == 0
            assert record.trigger_active_fraction == 0.0


@pytest.mark.parametrize(
    ("trigger", "expected"),
    [
        ([0, 0, 0], []),
        ([1, 1, 1], [(0, 3)]),
        ([0, 1, 1, 0, 0, 1, 0], [(1, 3), (5, 6)]),
        ([1, 0, 1], [(0, 1), (2, 3)]),
    ],
)
def test_trigger_run_splitting(trigger, expected):
    runs = trigger_runs(np.array(trigger, dtype=float))
    assert [(r.start_sample, r.end_sample) for r in runs] == expected


def test_missing_data_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path / "nope")
