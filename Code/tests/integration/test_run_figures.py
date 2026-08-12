"""A finished training run must explain itself: end-to-end figure generation.

Runs the real training CLI on synthetic participants and then checks the folder a reader
actually opens -- that every catalogued figure was attempted, that nothing failed silently,
that each written figure exists as both PNG and PDF, and that the index and README describe
what is there. Also checks the two escape hatches: ``--no-figures`` on the trainer, and
``bruxism-figures`` rebuilding the set afterwards without retraining.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bruxism.cli import make_run_figures, run_nested_loso
from bruxism.utils.io import write_yaml
from bruxism.visualization.run_figures import FIGURE_CATALOGUE

pytestmark = pytest.mark.slow


CONFIG = {
    "name": "figures_itest",
    "task_id": "five_class",
    "model_id": "dual_branch_wavelet_cnn",
    "modality": "fusion",
    "data": {
        "sampling_rate_hz": 1200,
        "window_seconds": 1.0,
        "stride_seconds": 0.5,
        "guard_seconds": 0.25,
        "startup_guard_seconds": 0.5,
        "cache_dir": None,
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 0.005,
        "num_workers": 0,
        "device": "cpu",
        "deterministic": True,
        "seeds": [0],
        "search_space": {},
    },
    "selection": {"min_epochs": 1, "max_epochs": 1, "patience": 1},
    "augmentation": {"enabled": True, "probability": 0.5},
    # The fixture microphone is clean, but the mic branch still reads A5 (0-18.75 Hz)
    # behind a 20 Hz high-pass, so assert_bands_are_inside_their_passband refuses the run
    # unless it is declared. Declared here rather than fixed because these tests exercise
    # the SHIPPED architecture; a test that quietly used different bands would stop
    # covering it.
    "stopband_bands_acknowledged_by": "integration test, 2026-08-12: exercises the shipped mic branch",
}


def _config(tmp_path: Path, root: Path) -> Path:
    payload = json.loads(json.dumps(CONFIG))
    payload["data"]["data_root"] = str(root)
    payload["output"] = {
        "runs_root": str(tmp_path / "runs"),
        "run_id": "figures_itest",
        "overwrite": True,
    }
    path = tmp_path / "config.yaml"
    write_yaml(path, payload)
    return path


def _train(config_path: Path, root: Path, *extra: str) -> int:
    return run_nested_loso.main(
        [
            "--config",
            str(config_path),
            "--data-root",
            str(root),
            "--max-folds",
            "1",
            "--no-resume",
            "--quiet",
            *extra,
        ]
    )


def _index(run_dir: Path) -> dict:
    return json.loads((run_dir / "figures" / "figure_index.json").read_text())


def test_a_finished_run_writes_its_own_figures(synthetic_root, tmp_path):
    config_path = _config(tmp_path, synthetic_root)
    assert _train(config_path, synthetic_root) == 0

    run_dir = tmp_path / "runs" / "figures_itest"
    figures = run_dir / "figures"
    index = _index(run_dir)

    # Every catalogued figure is accounted for, one way or another.
    reported = {record["stem"] for record in index["figures"]}
    assert reported == set(FIGURE_CATALOGUE)

    failures = [r for r in index["figures"] if r["status"] == "failed"]
    assert not failures, f"figures failed: {[(r['stem'], r['reason']) for r in failures]}"

    for record in index["figures"]:
        assert record["status"] in {"written", "skipped"}
        if record["status"] == "skipped":
            # A missing figure must say why, so the gap is never mistaken for an oversight.
            assert record["reason"], record["stem"]
            continue
        for suffix in (".png", ".pdf"):
            path = figures / f"{record['stem']}{suffix}"
            assert path.is_file(), f"{path} missing"
            assert path.stat().st_size > 1000

    # Provenance is carried into the index, so a copied folder still names its run.
    assert index["run_id"] == "figures_itest"
    assert index["config_hash"] and index["manifest_hash"]
    assert index["results_condition"]["task_id"] == "five_class"
    assert index["counts"]["written"] >= 15

    readme = (figures / "README.md").read_text()
    assert "figure_index.json" in readme
    assert "bruxism-figures" in readme
    # The scope caveat travels with the figures.
    assert "instructed" in readme
    assert "clinical bruxism-detection" in readme
    assert "instructed grinding" in readme
    for record in index["figures"]:
        assert record["stem"] in readme


def test_no_figures_leaves_the_folder_empty_but_the_bundle_complete(synthetic_root, tmp_path):
    config_path = _config(tmp_path, synthetic_root)
    assert _train(config_path, synthetic_root, "--no-figures") == 0

    run_dir = tmp_path / "runs" / "figures_itest"
    assert (run_dir / "predictions.parquet").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert not list((run_dir / "figures").glob("*"))


def test_bruxism_figures_rebuilds_the_set_without_retraining(synthetic_root, tmp_path):
    config_path = _config(tmp_path, synthetic_root)
    assert _train(config_path, synthetic_root, "--no-figures") == 0
    run_dir = tmp_path / "runs" / "figures_itest"
    checkpoints = sorted(path.stat().st_mtime_ns for path in (run_dir / "checkpoints").glob("*.pt"))

    assert (
        make_run_figures.main(
            ["--run-dir", str(run_dir), "--data-root", str(synthetic_root), "--quiet"]
        )
        == 0
    )
    index = _index(run_dir)
    assert index["counts"]["written"] >= 15
    assert not [r for r in index["figures"] if r["status"] == "failed"]
    # Nothing was retrained: the checkpoints are untouched.
    assert (
        sorted(path.stat().st_mtime_ns for path in (run_dir / "checkpoints").glob("*.pt"))
        == checkpoints
    )


def test_figures_are_skipped_with_a_reason_when_the_data_root_is_absent(synthetic_root, tmp_path):
    """Without raw data the signal-side figures cannot be drawn -- and must say so."""
    config_path = _config(tmp_path, synthetic_root)
    assert _train(config_path, synthetic_root, "--no-figures") == 0
    run_dir = tmp_path / "runs" / "figures_itest"

    # Point the stored config at a data root that no longer exists, and supply none.
    resolved = (run_dir / "resolved_config.yaml").read_text()
    (run_dir / "resolved_config.yaml").write_text(
        resolved.replace(str(synthetic_root), str(tmp_path / "gone"))
    )
    assert make_run_figures.main(["--run-dir", str(run_dir), "--quiet"]) == 0

    index = _index(run_dir)
    skipped = {r["stem"]: r["reason"] for r in index["figures"] if r["status"] == "skipped"}
    assert "01_dataset_inventory" in skipped
    assert "data" in skipped["01_dataset_inventory"]
    # The ledger-derived figures still work without any raw signal.
    written = {r["stem"] for r in index["figures"] if r["status"] == "written"}
    assert {"13_confusion_matrix", "18_participant_class_recall", "19_calibration"} <= written


def test_a_missing_run_directory_is_an_actionable_error(tmp_path, capsys):
    assert make_run_figures.main(["--run-dir", str(tmp_path / "nope"), "--quiet"]) == 2
    assert "run directory not found" in capsys.readouterr().err
