"""Figure helpers, and that every figure function actually writes a PNG and a PDF.

The drawing tests are deliberately shallow -- a figure's *appearance* is not testable -- but
they do assert the three properties that have broken silently before: the file pair is
written, the sampling that feeds a figure is deterministic, and the catalogue that the index
and README are generated from stays consistent with the code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bruxism.data.dataset import RecordingCache
from bruxism.data.labels import get_task
from bruxism.evaluation.metrics import LEDGER_COLUMNS, PredictionLedger
from bruxism.preprocessing.augmentation import AugmentationConfig
from bruxism.preprocessing.filters import FilterChainConfig
from bruxism.preprocessing.wavelets import WaveletConfig
from bruxism.visualization import diagnostics, signal_figures
from bruxism.visualization.run_figures import FIGURE_CATALOGUE

FIVE_CLASS = get_task("five_class")


@pytest.fixture
def cache(synthetic_manifest) -> RecordingCache:
    """A recording cache with disk caching off, so tests leave nothing behind."""
    return RecordingCache(synthetic_manifest, FilterChainConfig(), cache_dir=None)


def _pair_exists(paths, output_dir: Path, stem: str) -> None:
    assert {Path(path).suffix for path in paths} == {".png", ".pdf"}
    for suffix in (".png", ".pdf"):
        path = output_dir / f"{stem}{suffix}"
        assert path.is_file(), f"{path} was not written"
        assert path.stat().st_size > 1000, f"{path} is suspiciously small"


# ------------------------------------------------------------------- helpers ---


def test_sample_windows_by_class_is_deterministic_and_covers_every_class(synthetic_window_index):
    first = signal_figures.sample_windows_by_class(
        synthetic_window_index, FIVE_CLASS, max_per_class=7
    )
    second = signal_figures.sample_windows_by_class(
        synthetic_window_index, FIVE_CLASS, max_per_class=7
    )
    assert set(first) == set(range(FIVE_CLASS.num_classes))
    assert all(len(windows) <= 7 for windows in first.values())
    # No RNG anywhere: the same index must yield the identical example ids.
    assert {label: [w.sample_id for w in windows] for label, windows in first.items()} == {
        label: [w.sample_id for w in windows] for label, windows in second.items()
    }


def test_sample_windows_by_class_omits_families_the_task_excludes(synthetic_window_index):
    no_chewing = get_task("no_chewing_four_class")
    sampled = signal_figures.sample_windows_by_class(
        synthetic_window_index, no_chewing, max_per_class=5
    )
    assert set(sampled) == set(range(no_chewing.num_classes))
    families = {window.task_family for windows in sampled.values() for window in windows}
    assert "chewing" not in families


def test_expected_calibration_error_is_zero_when_confidence_matches_accuracy():
    # Two bins, each internally consistent: 90% confident and 90% correct, 50/50.
    confidence = np.array([0.9] * 10 + [0.5] * 10)
    correct = np.array([1.0] * 9 + [0.0] + [1.0] * 5 + [0.0] * 5)
    ece, mean_confidence, accuracy, counts = diagnostics.expected_calibration_error(
        confidence, correct, bins=10
    )
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert counts.sum() == 20
    usable = ~np.isnan(accuracy)
    assert np.allclose(accuracy[usable], mean_confidence[usable])


def test_expected_calibration_error_detects_overconfidence():
    confidence = np.full(100, 0.95)
    correct = np.concatenate([np.ones(50), np.zeros(50)])
    ece, _, _, _ = diagnostics.expected_calibration_error(confidence, correct, bins=10)
    assert ece == pytest.approx(0.45, abs=1e-6)


def test_figure_catalogue_is_complete_and_describes_every_entry():
    assert len(FIGURE_CATALOGUE) == len(set(FIGURE_CATALOGUE))
    for stem, (title, shows, slot) in FIGURE_CATALOGUE.items():
        assert stem[:2].isdigit(), f"{stem} must start with its ordering prefix"
        assert title and shows and slot, f"{stem} is missing a description"
        assert shows.endswith("."), f"{stem}: 'what it shows' should be a sentence"


# ------------------------------------------------------------ signal figures ---


def test_dataset_inventory_and_window_inventory_write_both_formats(
    synthetic_window_index, tmp_path
):
    _pair_exists(
        signal_figures.plot_dataset_inventory(synthetic_window_index, FIVE_CLASS, tmp_path),
        tmp_path,
        "01_dataset_inventory",
    )
    _pair_exists(
        signal_figures.plot_window_inventory(synthetic_window_index, FIVE_CLASS, tmp_path),
        tmp_path,
        "02_window_inventory",
    )


def test_filter_response_reports_the_zero_phase_mode(tmp_path):
    _pair_exists(
        signal_figures.plot_filter_response(FilterChainConfig(), 1200.0, tmp_path),
        tmp_path,
        "04_filter_response",
    )


def test_segmentation_timeline_and_preprocessing_stages_use_real_recordings(
    synthetic_manifest, synthetic_window_index, cache, tmp_path
):
    _pair_exists(
        signal_figures.plot_segmentation_timeline(
            synthetic_manifest, synthetic_window_index, cache, tmp_path
        ),
        tmp_path,
        "03_segmentation_timeline",
    )
    _pair_exists(
        signal_figures.plot_preprocessing_stages(
            synthetic_manifest,
            synthetic_window_index,
            tmp_path,
            filter_config=FilterChainConfig(),
        ),
        tmp_path,
        "05_preprocessing_stages",
    )


def test_window_wavelet_and_augmentation_figures(synthetic_window_index, cache, tmp_path):
    emg = WaveletConfig(wavelet="db4", level=4, bands=("A4", "D3", "D1"))
    mic = WaveletConfig(wavelet="coif5", level=5, bands=("A5", "D3", "D1"))
    _pair_exists(
        signal_figures.plot_example_windows(
            synthetic_window_index, cache, FIVE_CLASS, tmp_path, overlay=6
        ),
        tmp_path,
        "07_example_windows",
    )
    _pair_exists(
        signal_figures.plot_class_spectra(
            synthetic_window_index, cache, FIVE_CLASS, tmp_path, max_per_class=8
        ),
        tmp_path,
        "06_class_spectra",
    )
    _pair_exists(
        signal_figures.plot_wavelet_bands(
            synthetic_window_index, cache, FIVE_CLASS, tmp_path, emg_wavelet=emg, mic_wavelet=mic
        ),
        tmp_path,
        "08_wavelet_bands",
    )
    _pair_exists(
        signal_figures.plot_wavelet_band_energy(
            synthetic_window_index,
            cache,
            FIVE_CLASS,
            tmp_path,
            emg_wavelet=emg,
            mic_wavelet=mic,
            max_per_class=6,
        ),
        tmp_path,
        "09_wavelet_band_energy",
    )
    _pair_exists(
        signal_figures.plot_augmentation_examples(
            synthetic_window_index, cache, FIVE_CLASS, AugmentationConfig(), tmp_path
        ),
        tmp_path,
        "10_augmentation_examples",
    )


# --------------------------------------------------------------- diagnostics ---


def _ledger(
    n_per_subject: int = 40, subjects: tuple[str, ...] = ("S01", "S02")
) -> PredictionLedger:
    """A small but valid prediction ledger over the five-class label space."""
    names = FIVE_CLASS.class_names
    rng = np.random.default_rng(7)
    rows: list[dict] = []
    for subject in subjects:
        for index in range(n_per_subject):
            true_label = index % len(names)
            probabilities = rng.dirichlet(np.full(len(names), 0.6))
            predicted = int(np.argmax(probabilities))
            rows.append(
                {
                    "sample_id": f"{subject}#{index:04d}",
                    "subject_id": subject,
                    "recording_id": f"{subject}_rec{index % 3}",
                    "start_sample": index * 600,
                    "end_sample": index * 600 + 1200,
                    "start_seconds": index * 0.5,
                    "end_seconds": index * 0.5 + 1.0,
                    "condition": f"cond{index % 3}",
                    "task_family": "rest",
                    "segment_source": "trigger_active",
                    "true_label": true_label,
                    "predicted_label": predicted,
                    "true_class": names[true_label],
                    "predicted_class": names[predicted],
                    "outer_fold": 0,
                    "seed": 0,
                    "task_id": "five_class",
                    "model_id": "dual_branch_wavelet_cnn",
                    "modality": "fusion",
                    "source_commit": "c",
                    "config_hash": "h",
                    "manifest_hash": "m",
                    "checkpoint_sha256": "s",
                    **{f"prob_{name}": float(probabilities[j]) for j, name in enumerate(names)},
                }
            )
    frame = pd.DataFrame(rows)
    for column in LEDGER_COLUMNS:
        assert column in frame.columns
    return PredictionLedger(frame=frame, class_names=names)


def test_calibration_participant_recall_and_error_timeline(tmp_path):
    ledger = _ledger()
    _pair_exists(diagnostics.plot_calibration(ledger, tmp_path), tmp_path, "19_calibration")
    _pair_exists(
        diagnostics.plot_participant_class_recall(ledger, tmp_path),
        tmp_path,
        "18_participant_class_recall",
    )
    _pair_exists(diagnostics.plot_error_timeline(ledger, tmp_path), tmp_path, "21_error_timeline")
    # The projection CSV lets the t-SNE be re-plotted without recomputing embeddings.
    embeddings = np.random.default_rng(3).normal(size=(120, 8))
    labels = np.arange(120) % FIVE_CLASS.num_classes
    subjects = np.array(["S01" if i % 2 else "S02" for i in range(120)])
    paths, settings = diagnostics.plot_embedding_projection(
        embeddings, labels, subjects, FIVE_CLASS.class_names, tmp_path, perplexity=10.0
    )
    _pair_exists(paths, tmp_path, "22_embedding_tsne")
    assert settings["status"] == "EXPLORATORY"
    assert settings["n_samples"] == 120
    assert (tmp_path / "22_embedding_tsne_projection.csv").is_file()


def test_training_and_selection_figures_read_fold_outcomes(tmp_path):
    outcomes = [
        {
            "outer_fold": fold,
            "seed": seed,
            "test_subject": f"S0{fold + 1}",
            "epoch_budget": 3 + fold,
            "history_scope": "refit_training_fit",
            "selected_hyperparameters": {"learning_rate": 0.001},
            "training_history": [
                {
                    "epoch": epoch,
                    "train_loss": 1.0 / epoch,
                    "val_loss": 1.1 / epoch,
                    "val_macro_f1": 0.4 + 0.05 * epoch,
                }
                for epoch in range(1, 4)
            ],
            "inner_trials": [
                {
                    "trial_id": f"trial{index:02d}",
                    "hyperparameters": {"learning_rate": rate},
                    "mean_objective": 0.5 + 0.1 * index,
                    "std_objective": 0.05,
                    "failed": False,
                    "inner_selections": [
                        {"best_epoch": 2, "best_value": 0.5, "objective": "macro_f1"}
                    ],
                }
                for index, rate in enumerate((0.0003, 0.001))
            ],
        }
        for fold in range(2)
        for seed in (0, 1)
    ]
    _pair_exists(
        diagnostics.plot_training_curves_by_seed(outcomes, tmp_path), tmp_path, "11_training_curves"
    )
    _pair_exists(
        diagnostics.plot_hyperparameter_selection(outcomes, tmp_path),
        tmp_path,
        "12_hyperparameter_selection",
    )


def test_seed_stability_refuses_a_single_seed(tmp_path):
    metrics = {
        "conditions": {
            "five_class::m::fusion::seed0": {
                "task_id": "five_class",
                "seed": 0,
                "subject_level": {"per_subject": {"S01": {"macro_f1": 0.4}}},
            }
        }
    }
    with pytest.raises(ValueError, match="at least two"):
        diagnostics.plot_seed_stability(metrics, "five_class", tmp_path)
