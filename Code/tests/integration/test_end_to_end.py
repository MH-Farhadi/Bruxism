"""End-to-end integration on synthetic participants only.

These tests exercise the real CLIs and the real training engine, so they catch wiring
mistakes that unit tests cannot -- but they never touch the private data root.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bruxism.cli import audit_dataset, make_paper_artifacts, run_nested_loso, summarize_runs
from bruxism.config import ConfigError, ExperimentConfig, load_experiment_config
from bruxism.evaluation.metrics import PredictionLedger
from bruxism.utils.io import write_yaml

pytestmark = pytest.mark.slow


SMOKE_CONFIG = {
    "name": "test_smoke",
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
    "selection": {"min_epochs": 1, "max_epochs": 2, "patience": 1},
    "augmentation": {"enabled": True, "probability": 0.5},
    # The fixture microphone is clean, but the mic branch still reads A5 (0-18.75 Hz)
    # behind a 20 Hz high-pass, so assert_bands_are_inside_their_passband refuses the run
    # unless it is declared. Declared here rather than fixed because these tests exercise
    # the SHIPPED architecture; a test that quietly used different bands would stop
    # covering it.
    "stopband_bands_acknowledged_by": "integration test, 2026-08-12: exercises the shipped mic branch",
}


def _write_config(tmp_path: Path, root: Path, **overrides) -> Path:
    payload = json.loads(json.dumps(SMOKE_CONFIG))
    payload["data"]["data_root"] = str(root)
    payload["output"] = {"runs_root": str(tmp_path / "runs"), "run_id": "itest", "overwrite": True}
    payload.update(overrides)
    path = tmp_path / "config.yaml"
    write_yaml(path, payload)
    return path


def test_audit_cli_produces_a_complete_bundle(synthetic_root, tmp_path, capsys):
    out = tmp_path / "audit"
    code = audit_dataset.main(
        ["--data-root", str(synthetic_root), "--output-root", str(out), "--no-video", "--quiet"]
    )
    assert code == 0
    bundle = next(out.iterdir())
    for name in (
        "manifest.parquet",
        "manifest.csv",
        "data_audit.json",
        "data_audit.md",
        "trigger_summary.csv",
        "guard_sensitivity.csv",
        "window_counts.csv",
    ):
        assert (bundle / name).is_file(), name

    audit = json.loads((bundle / "data_audit.json").read_text())
    assert audit["totals"]["n_csv"] == 60
    assert "metadata_condition_conflict" in audit["quality_flags"]
    assert audit["signal_units"] == "arbitrary_adc_units"


def test_audit_artifacts_contain_no_identifiers_or_absolute_paths(synthetic_root, tmp_path):
    out = tmp_path / "audit"
    audit_dataset.main(
        [
            "--data-root",
            str(synthetic_root),
            "--output-root",
            str(out),
            "--no-video",
            "--no-figures",
            "--quiet",
        ]
    )
    forbidden = ("Survey", "Receipt", "Reconciliation", "IMG_", ".HEIC", "/home/", "C:\\")
    for path in out.rglob("*"):
        if path.suffix in {".json", ".csv", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in forbidden:
                assert token not in text, f"{path.name} leaked {token!r}"


def test_full_nested_loso_smoke_run(synthetic_root, tmp_path):
    config_path = _write_config(tmp_path, synthetic_root)
    assert (
        run_nested_loso.main(
            [
                "--config",
                str(config_path),
                "--data-root",
                str(synthetic_root),
                "--max-folds",
                "2",
                "--no-figures",
            ]
        )
        == 0
    )

    run_dir = tmp_path / "runs" / "itest"
    for name in (
        "resolved_config.yaml",
        "environment.json",
        "data_manifest.json",
        "data_manifest.sha256",
        "source_state.json",
        "folds.json",
        "predictions.parquet",
        "metrics.json",
        "metrics.csv",
    ):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "selection" / "fold_outcomes.json").is_file()
    assert list((run_dir / "checkpoints").glob("*.pt"))
    assert (run_dir / "logs" / "run.log.jsonl").is_file()

    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    ledger = PredictionLedger(
        frame=predictions,
        class_names=("rest", "movement", "clench", "instructed_grinding", "chewing"),
    )
    ledger.assert_exactly_once()

    # Each executed fold contributes exactly one held-out participant.
    assert predictions.groupby("outer_fold")["subject_id"].nunique().eq(1).all()
    # Provenance travels with every row.
    for column in ("config_hash", "manifest_hash", "checkpoint_sha256", "window_index_hash"):
        assert predictions[column].notna().all()
    assert predictions["safe_for_inference"].all()

    outcomes = json.loads((run_dir / "selection" / "fold_outcomes.json").read_text())
    for outcome in outcomes:
        # The normalizer never saw the held-out participant.
        assert outcome["test_subject"] not in outcome["normalizer"]["fitted_on"]
        assert len(outcome["normalizer"]["fitted_on"]) == 4
        # Inner LOSO had four folds, not five.
        for trial in outcome["inner_trials"]:
            assert len(trial["inner_selections"]) == 4
            assert outcome["test_subject"] not in trial["inner_fold_subjects"]


def test_resume_reuses_completed_folds(synthetic_root, tmp_path):
    config_path = _write_config(tmp_path, synthetic_root)
    args = [
        "--config",
        str(config_path),
        "--data-root",
        str(synthetic_root),
        "--max-folds",
        "1",
        "--no-figures",
    ]
    assert run_nested_loso.main(args) == 0
    first = (tmp_path / "runs" / "itest" / "predictions.parquet").read_bytes()
    assert run_nested_loso.main(args) == 0
    assert (tmp_path / "runs" / "itest" / "predictions.parquet").read_bytes() == first


def test_resume_refuses_an_incompatible_configuration(synthetic_root, tmp_path):
    config_path = _write_config(tmp_path, synthetic_root)
    run_nested_loso.main(
        ["--config", str(config_path), "--data-root", str(synthetic_root), "--max-folds", "1"]
    )
    # Changing the guard changes the window index -> the stored fold is not reusable.
    code = run_nested_loso.main(
        [
            "--config",
            str(config_path),
            "--data-root",
            str(synthetic_root),
            "--max-folds",
            "1",
            "--set",
            "data.guard_seconds=0.4",
        ]
    )
    assert code == 2, "resuming across an incompatible window index must fail loudly"


def test_validate_only_writes_a_plan_without_training(synthetic_root, tmp_path):
    config_path = _write_config(tmp_path, synthetic_root)
    assert (
        run_nested_loso.main(
            [
                "--config",
                str(config_path),
                "--data-root",
                str(synthetic_root),
                "--validate-only",
                "--no-figures",
            ]
        )
        == 0
    )
    run_dir = tmp_path / "runs" / "itest"
    plan = json.loads((run_dir / "folds.json").read_text())
    assert plan["folds"]["n_outer_folds"] == 5
    assert plan["folds"]["n_inner_folds_per_outer"] == 4
    assert not (run_dir / "predictions.parquet").exists()


def _log_messages(run_dir: Path) -> list[str]:
    lines = (run_dir / "logs" / "run.log.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["message"] for line in lines]


def _log_loggers(run_dir: Path) -> set[str]:
    lines = (run_dir / "logs" / "run.log.jsonl").read_text(encoding="utf-8").splitlines()
    return {json.loads(line)["logger"] for line in lines}


def _one_fast_fold(config_path: Path, root: Path, *extra: str) -> list[str]:
    """CLI arguments for the cheapest run that still exercises every training stage.

    Figures are switched off: they are covered by ``test_run_figures.py`` and drawing them
    would add seconds to every test here that is about the run bundle instead.
    """
    return [
        "--config",
        str(config_path),
        "--data-root",
        str(root),
        "--max-folds",
        "1",
        "--no-resume",
        "--no-figures",
        "--set",
        "selection.max_epochs=1",
        *extra,
    ]


def test_a_training_run_reports_its_progress(synthetic_root, tmp_path):
    """A run that takes hours must say what it is doing while it does it."""
    config_path = _write_config(tmp_path, synthetic_root)
    args = _one_fast_fold(
        config_path, synthetic_root, "--progress", "plain", "--progress-interval", "0"
    )
    assert run_nested_loso.main(args) == 0
    messages = _log_messages(tmp_path / "runs" / "itest")

    # The size of the job is stated before any of it runs.
    assert any(text.startswith("plan: ") and "fold run(s)" in text for text in messages)
    # Every stage announces itself, with the numbers a reader needs to judge it.
    assert any("inner search: 1 trial(s) x 4 inner fold(s)" in text for text in messages)
    assert any("best val_macro_f1=" in text and "at epoch" in text for text in messages)
    assert any("refitting on" in text and "for exactly" in text for text in messages)
    assert any("accuracy=" in text and "macro_f1=" in text for text in messages)
    assert any("remaining" in text for text in messages)
    # Per-epoch ticks arrive between those milestones rather than dead air.
    assert any("epoch (" in text and "elapsed" in text for text in messages)


def test_progress_display_does_not_change_what_a_run_produces(synthetic_root, tmp_path):
    """``--progress`` is display only: it must not touch identity, splits or the ledger.

    Both runs write the same run directory, because the output paths are part of the
    configuration hash -- two runs pointed at different directories could never be
    compared on identity in the first place.
    """
    config_path = _write_config(tmp_path, synthetic_root)
    run_dir = tmp_path / "runs" / "itest"

    assert (
        run_nested_loso.main(_one_fast_fold(config_path, synthetic_root, "--progress", "none")) == 0
    )
    quiet = pd.read_parquet(run_dir / "predictions.parquet")
    quiet_bundle = json.loads((run_dir / "run_bundle.json").read_text())
    quiet_loggers = _log_loggers(run_dir)

    assert (
        run_nested_loso.main(
            _one_fast_fold(
                config_path, synthetic_root, "--progress", "plain", "--progress-interval", "0"
            )
        )
        == 0
    )
    loud = pd.read_parquet(run_dir / "predictions.parquet")
    loud_bundle = json.loads((run_dir / "run_bundle.json").read_text())

    for key in ("config_hash", "manifest_hash", "window_index_hash"):
        assert quiet_bundle[key] == loud_bundle[key], key
    # Identical windows, identical fold assignment, identical held-out participant.
    for column in ("sample_id", "subject_id", "true_label", "outer_fold", "seed"):
        assert quiet[column].tolist() == loud[column].tolist(), column

    # Only the display differs: the reporter is silent under --progress none. The run log
    # is appended to, so the second read covers both runs and only needs the new records.
    assert "bruxism.utils.progress" not in quiet_loggers
    assert "bruxism.utils.progress" in _log_loggers(run_dir) - quiet_loggers


def test_summarize_and_report_regenerate_everything_from_artifacts(synthetic_root, tmp_path):
    config_path = _write_config(tmp_path, synthetic_root)
    run_nested_loso.main(
        ["--config", str(config_path), "--data-root", str(synthetic_root), "--max-folds", "2"]
    )
    runs_root = tmp_path / "runs"
    assert (
        summarize_runs.main(
            ["--runs-root", str(runs_root), "--output", str(tmp_path / "summary"), "--quiet"]
        )
        == 0
    )

    bundle = tmp_path / "bundle"
    assert (
        make_paper_artifacts.main(
            ["--runs-root", str(runs_root), "--output-root", str(bundle), "--no-tsne", "--quiet"]
        )
        == 0
    )
    assert (bundle / "paper_results.md").is_file()
    assert (bundle / "metrics.json").is_file()
    assert list((bundle / "figures").glob("*.png"))
    assert list((bundle / "tables").glob("*.tex"))

    report = (bundle / "paper_results.md").read_text()
    assert "instructed" in report
    assert "BLOCKED" in report, "unavailable analyses must be named, not silently omitted"
    assert "/home/" not in report

    macros = (bundle / "tables" / "macros.tex").read_text()
    assert r"\newcommand{\brux" in macros


def test_report_figures_come_only_from_saved_artifacts(synthetic_root, tmp_path):
    """The report must run with the data root unavailable (t-SNE excepted)."""
    config_path = _write_config(tmp_path, synthetic_root)
    run_nested_loso.main(
        ["--config", str(config_path), "--data-root", str(synthetic_root), "--max-folds", "1"]
    )
    bundle = tmp_path / "bundle_nodata"
    assert (
        make_paper_artifacts.main(
            ["--runs-root", str(tmp_path / "runs"), "--output-root", str(bundle), "--quiet"]
        )
        == 0
    )
    assert list((bundle / "figures").glob("*confusion_matrix.png"))
    report = (bundle / "paper_results.md").read_text()
    assert "t-SNE" in report and "data-root" in report


# ------------------------------------------------------------------ config ---


def test_unknown_config_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown top-level config key"):
        ExperimentConfig.from_dict({"name": "x", "nonsense": 1})
    with pytest.raises(ConfigError, match="unknown key"):
        ExperimentConfig.from_dict({"name": "x", "training": {"batch_sise": 4}})


def test_unknown_task_id_is_rejected():
    with pytest.raises(KeyError, match="unknown task_id"):
        ExperimentConfig.from_dict({"name": "x", "task_id": "seven_class"})


def test_shipped_configs_all_load_and_hash_distinctly():
    paths = sorted(Path("configs/experiments").glob("*.yaml"))
    assert paths
    hashes = {}
    for path in paths:
        config = load_experiment_config(path)
        hashes[config.name] = config.config_hash
    assert len(set(hashes.values())) == len(hashes)


def test_config_hash_changes_with_any_meaningful_field():
    base = ExperimentConfig.from_dict({"name": "x"})
    from dataclasses import replace

    assert replace(base, task_id="ternary").config_hash != base.config_hash
    assert replace(base, modality="emg_only").config_hash != base.config_hash
    assert (
        replace(base, training=base.training.replace(learning_rate=0.5)).config_hash
        != base.config_hash
    )
