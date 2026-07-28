"""Metrics, compared against fixed reference examples and hand computations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from bruxism.evaluation.metrics import (
    LEDGER_COLUMNS,
    PredictionLedger,
    binary_metrics,
    confusion_matrices,
    curve_points,
    one_vs_rest_auc,
    pooled_window_metrics,
    subject_level_summary,
)


def make_ledger(rows, class_names):
    """Build a valid ledger from a compact row spec."""
    frame = pd.DataFrame(rows)
    for column in LEDGER_COLUMNS:
        if column not in frame.columns:
            frame[column] = (
                "x" if not column.endswith(("_sample", "_seconds", "label", "fold", "seed")) else 0
            )
    return PredictionLedger(frame=frame, class_names=tuple(class_names))


def _rows(subject, y_true, y_pred, probabilities, class_names):
    return [
        {
            "sample_id": f"{subject}#{i}",
            "subject_id": subject,
            "recording_id": f"{subject}_r",
            "start_sample": i,
            "end_sample": i + 1200,
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "true_label": int(t),
            "predicted_label": int(p),
            "true_class": class_names[t],
            "predicted_class": class_names[p],
            "outer_fold": 0,
            "seed": 0,
            "task_id": "t",
            "model_id": "m",
            "modality": "fusion",
            "source_commit": "c",
            "config_hash": "h",
            "manifest_hash": "mh",
            "checkpoint_sha256": "s",
            **{f"prob_{n}": float(probabilities[i][j]) for j, n in enumerate(class_names)},
        }
        for i, (t, p) in enumerate(zip(y_true, y_pred))
    ]


# ------------------------------------------------------------------ ledger ---


def test_ledger_requires_every_provenance_column():
    frame = pd.DataFrame({"sample_id": ["a"], "prob_x": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        PredictionLedger(frame=frame, class_names=("x",))


def test_ledger_rejects_probabilities_that_do_not_sum_to_one():
    names = ["a", "b"]
    rows = _rows("S01", [0, 1], [0, 1], [[0.5, 0.2], [0.3, 0.7]], names)
    with pytest.raises(ValueError, match="do not sum to 1"):
        make_ledger(rows, names)


def test_duplicate_predictions_are_detected():
    names = ["a", "b"]
    rows = _rows("S01", [0, 1], [0, 1], [[1.0, 0.0], [0.0, 1.0]], names)
    rows.append({**rows[0]})
    ledger = make_ledger(rows, names)
    with pytest.raises(AssertionError, match="duplicate prediction rows"):
        ledger.assert_exactly_once()


def test_missing_predictions_are_detected():
    names = ["a", "b"]
    ledger = make_ledger(_rows("S01", [0], [0], [[1.0, 0.0]], names), names)
    with pytest.raises(AssertionError, match="have no prediction"):
        ledger.assert_covers(["S01#0", "S01#1"])


# ------------------------------------------------------- confusion matrices ---


def test_confusion_matrices_against_a_fixed_example():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    matrices = confusion_matrices(y_true, y_pred, 3)
    assert matrices["raw"].tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert np.allclose(
        matrices["row_normalized"], [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]
    )
    assert matrices["raw"].sum() == len(y_true)


def test_absent_class_row_normalises_to_zero_not_nan():
    matrices = confusion_matrices(np.array([0, 0]), np.array([0, 1]), 3)
    assert np.isfinite(matrices["row_normalized"]).all()
    assert matrices["row_normalized"][2].tolist() == [0.0, 0.0, 0.0]
    assert matrices["raw"][2].sum() == 0


# ------------------------------------------------------------------- AUC ---


def test_one_vs_rest_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 200)
    scores = rng.random((200, 3))
    scores /= scores.sum(axis=1, keepdims=True)
    result = one_vs_rest_auc(y_true, scores, ["a", "b", "c"])
    for index, name in enumerate(["a", "b", "c"]):
        assert result["roc_auc_per_class"][name] == pytest.approx(
            roc_auc_score((y_true == index).astype(int), scores[:, index])
        )
        assert result["average_precision_per_class"][name] == pytest.approx(
            average_precision_score((y_true == index).astype(int), scores[:, index])
        )


def test_absent_class_gets_no_fabricated_auc():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.3, 0.6, 0.1]])
    result = one_vs_rest_auc(y_true, scores, ["a", "b", "c"])
    assert result["roc_auc_per_class"]["c"] is None
    assert result["average_precision_per_class"]["c"] is None
    assert result["classes_without_auc"] == ["c"]
    assert result["n_classes_averaged"] == 2
    assert result["n_classes_total"] == 3
    # The macro average is over the two evaluable classes only, and says so.
    assert result["macro_roc_auc"] == pytest.approx(
        np.mean([result["roc_auc_per_class"]["a"], result["roc_auc_per_class"]["b"]])
    )


def test_perfect_and_inverted_separations():
    y_true = np.array([0, 0, 1, 1])
    perfect = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    inverted = perfect[:, ::-1]
    assert one_vs_rest_auc(y_true, perfect, ["a", "b"])["macro_roc_auc"] == 1.0
    assert one_vs_rest_auc(y_true, inverted, ["a", "b"])["macro_roc_auc"] == 0.0


# ---------------------------------------------------------------- binary ---


def test_binary_metrics_against_a_hand_computed_confusion_matrix():
    # TP=3, FN=1, TN=4, FP=2
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
    scores = np.column_stack([1 - y_pred * 0.9 - 0.05, y_pred * 0.9 + 0.05])
    result = binary_metrics(y_true, y_pred, scores, positive_index=1)
    assert (result["true_positive"], result["false_negative"]) == (3, 1)
    assert (result["true_negative"], result["false_positive"]) == (4, 2)
    assert result["sensitivity_recall"] == pytest.approx(3 / 4)
    assert result["specificity"] == pytest.approx(4 / 6)
    assert result["ppv_precision"] == pytest.approx(3 / 5)
    assert result["npv"] == pytest.approx(4 / 5)
    assert result["f1"] == pytest.approx(2 * (3 / 4) * (3 / 5) / ((3 / 4) + (3 / 5)))
    assert result["positive_prevalence"] == pytest.approx(0.4)


def test_binary_metrics_on_a_degenerate_single_class_set():
    y = np.zeros(5, dtype=int)
    scores = np.column_stack([np.ones(5), np.zeros(5)])
    result = binary_metrics(y, y, scores, positive_index=1)
    assert result["roc_auc"] is None and result["pr_auc_average_precision"] is None
    assert result["sensitivity_recall"] is None  # no positives at all


# ------------------------------------------------------------ aggregation ---


def test_pooled_metrics_are_labelled_descriptive():
    names = ["a", "b"]
    ledger = make_ledger(
        _rows(
            "S01",
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]],
            names,
        ),
        names,
    )
    metrics = pooled_window_metrics(ledger)
    assert metrics["interpretation"] == "descriptive_only"
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["per_class"]["a"]["support"] == 2
    assert "binary" in metrics


def test_subject_level_is_primary_and_reports_every_participant():
    names = ["a", "b"]
    rows = _rows("S01", [0, 1], [0, 1], [[0.9, 0.1], [0.1, 0.9]], names)
    rows += _rows("S02", [0, 1], [1, 1], [[0.4, 0.6], [0.1, 0.9]], names)
    ledger = make_ledger(rows, names)
    summary = subject_level_summary(ledger)
    assert summary["interpretation"] == "primary"
    assert summary["unit_of_generalization"] == "participant"
    assert set(summary["per_subject"]) == {"S01", "S02"}
    assert summary["accuracy"]["values"] == {"S01": 1.0, "S02": 0.5}
    assert summary["accuracy"]["mean"] == pytest.approx(0.75)
    assert summary["accuracy"]["min"] == 0.5 and summary["accuracy"]["max"] == 1.0
    assert "population" in summary["note"]


def test_metrics_are_recomputable_from_the_ledger_alone():
    """The published summary must follow from predictions.parquet and nothing else."""
    names = ["a", "b", "c"]
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 3, 60)
    y_pred = rng.integers(0, 3, 60)
    probabilities = rng.random((60, 3))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    ledger = make_ledger(_rows("S01", y_true, y_pred, probabilities, names), names)

    first = pooled_window_metrics(ledger)
    roundtrip = PredictionLedger(frame=ledger.frame.copy(), class_names=ledger.class_names)
    assert pooled_window_metrics(roundtrip) == first


def test_curve_points_are_thinned_but_auc_is_not():
    names = ["a", "b"]
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, 400)
    probabilities = rng.random((400, 2))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    ledger = make_ledger(_rows("S01", y_true, y_true, probabilities, names), names)
    curves = curve_points(ledger, max_points=32)
    assert curves["a"]["available"] and len(curves["a"]["fpr"]) <= 32
    assert curves["a"]["roc_auc"] == pytest.approx(
        roc_auc_score((y_true == 0).astype(int), probabilities[:, 0])
    )
