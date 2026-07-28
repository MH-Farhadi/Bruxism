"""Metrics computed from a saved prediction ledger.

Every published number in this project is derived from :class:`PredictionLedger` -- a table
in which each held-out example appears exactly once per task / model / modality / seed, with
its true label, predicted label and full probability vector. Nothing is computed from
transient in-memory state during training and then reported.

Two conventions matter for honesty at this sample size:

* **Subject-level metrics are primary.** :func:`subject_level_summary` computes the metric
  within each participant and then summarises across participants, because participants --
  not windows -- are the units of generalisation. Windows inside one participant and one
  recording are strongly correlated.
* **Pooled-window metrics are secondary and labelled descriptive.**
  :func:`pooled_window_metrics` returns them with ``"interpretation": "descriptive_only"``
  attached, and the report generator prints that label next to the number.

An absent class in a fold is handled explicitly: its one-vs-rest AUC is ``None``, never a
fabricated 0.5 or 1.0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from bruxism.utils.logging import get_logger

__all__ = [
    "LEDGER_COLUMNS",
    "PredictionLedger",
    "binary_metrics",
    "confusion_matrices",
    "curve_points",
    "one_vs_rest_auc",
    "pooled_window_metrics",
    "subject_level_summary",
]

logger = get_logger(__name__)

#: Columns every prediction ledger must carry. Provenance columns are not optional: a
#: metric that cannot name its config, manifest and checkpoint is not reproducible.
LEDGER_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "subject_id",
    "recording_id",
    "start_sample",
    "end_sample",
    "start_seconds",
    "end_seconds",
    "true_label",
    "predicted_label",
    "true_class",
    "predicted_class",
    "outer_fold",
    "seed",
    "task_id",
    "model_id",
    "modality",
    "source_commit",
    "config_hash",
    "manifest_hash",
    "checkpoint_sha256",
)


@dataclass
class PredictionLedger:
    """A validated table of held-out predictions with per-class probabilities.

    Probability columns are named ``prob_<class_name>`` and must sum to 1 per row.
    """

    frame: pd.DataFrame
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        missing = [column for column in LEDGER_COLUMNS if column not in self.frame.columns]
        if missing:
            raise ValueError(f"prediction ledger is missing required columns: {missing}")
        probability_columns = self.probability_columns
        absent = [column for column in probability_columns if column not in self.frame.columns]
        if absent:
            raise ValueError(f"prediction ledger is missing probability columns: {absent}")
        if self.frame.empty:
            return
        totals = self.frame[list(probability_columns)].to_numpy(dtype=np.float64).sum(axis=1)
        if not np.allclose(totals, 1.0, atol=1e-4):
            worst = float(np.abs(totals - 1.0).max())
            raise ValueError(
                f"probability rows do not sum to 1 (max deviation {worst:.2e}); the model "
                f"must emit logits and the ledger must store softmax probabilities"
            )

    @property
    def probability_columns(self) -> tuple[str, ...]:
        return tuple(f"prob_{name}" for name in self.class_names)

    @property
    def y_true(self) -> np.ndarray:
        return self.frame["true_label"].to_numpy(dtype=np.int64)

    @property
    def y_pred(self) -> np.ndarray:
        return self.frame["predicted_label"].to_numpy(dtype=np.int64)

    @property
    def y_score(self) -> np.ndarray:
        return self.frame[list(self.probability_columns)].to_numpy(dtype=np.float64)

    @property
    def subjects(self) -> np.ndarray:
        return self.frame["subject_id"].to_numpy()

    def filter(self, **equals: Any) -> PredictionLedger:
        """Restrict to rows matching every ``column=value`` pair."""
        mask = pd.Series(True, index=self.frame.index)
        for column, value in equals.items():
            mask &= self.frame[column] == value
        return PredictionLedger(
            frame=self.frame[mask].reset_index(drop=True), class_names=self.class_names
        )

    def assert_exactly_once(
        self, *, by: Sequence[str] = ("task_id", "model_id", "modality", "seed")
    ) -> None:
        """Assert every sample appears exactly once within each configuration group.

        Raises
        ------
        AssertionError
            Naming the duplicated sample ids. A duplicate means an outer participant was
            evaluated twice, or two folds claimed the same window.
        """
        if self.frame.empty:
            return
        duplicated = self.frame.duplicated(subset=[*by, "sample_id"], keep=False)
        if duplicated.any():
            offenders = self.frame.loc[duplicated, "sample_id"].unique()[:5]
            raise AssertionError(
                f"{int(duplicated.sum())} duplicate prediction rows within "
                f"{list(by)} (e.g. {list(offenders)}); every held-out example must be "
                f"predicted exactly once per configuration"
            )

    def assert_covers(self, expected_sample_ids: Sequence[str]) -> None:
        """Assert no held-out example is missing from the ledger."""
        missing = set(expected_sample_ids) - set(self.frame["sample_id"])
        if missing:
            raise AssertionError(
                f"{len(missing)} held-out examples have no prediction (e.g. {sorted(missing)[:3]})"
            )


def confusion_matrices(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> dict[str, np.ndarray]:
    """Raw-count and row-normalised confusion matrices.

    Rows are true classes. A row whose support is zero normalises to zeros, not NaN, and
    the raw matrix shows the zero support.
    """
    raw = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    support = raw.sum(axis=1, keepdims=True)
    normalized = np.divide(
        raw, support, out=np.zeros(raw.shape, dtype=np.float64), where=support > 0
    )
    return {"raw": raw, "row_normalized": normalized}


def one_vs_rest_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """One-vs-rest ROC-AUC and average precision per class, plus macro averages.

    A class with no positive or no negative example in this subset gets ``None`` for both
    metrics and is excluded from the macro average, which is reported alongside the count
    of classes actually averaged. Nothing is imputed.
    """
    roc: dict[str, float | None] = {}
    average_precision: dict[str, float | None] = {}
    for index, name in enumerate(class_names):
        positive = (y_true == index).astype(np.int64)
        if positive.sum() == 0 or positive.sum() == positive.size:
            roc[name] = None
            average_precision[name] = None
            continue
        roc[name] = float(roc_auc_score(positive, y_score[:, index]))
        average_precision[name] = float(average_precision_score(positive, y_score[:, index]))

    usable_roc = [value for value in roc.values() if value is not None]
    usable_ap = [value for value in average_precision.values() if value is not None]
    return {
        "roc_auc_per_class": roc,
        "average_precision_per_class": average_precision,
        "macro_roc_auc": float(np.mean(usable_roc)) if usable_roc else None,
        "macro_average_precision": float(np.mean(usable_ap)) if usable_ap else None,
        "n_classes_averaged": len(usable_roc),
        "n_classes_total": len(class_names),
        "classes_without_auc": [name for name, value in roc.items() if value is None],
    }


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    positive_index: int = 1,
) -> dict[str, Any]:
    """Sensitivity, specificity, PPV, NPV, F1, ROC-AUC and PR-AUC for a binary task."""
    positive = (y_true == positive_index).astype(np.int64)
    predicted = (y_pred == positive_index).astype(np.int64)
    tp = int(((positive == 1) & (predicted == 1)).sum())
    tn = int(((positive == 0) & (predicted == 0)).sum())
    fp = int(((positive == 0) & (predicted == 1)).sum())
    fn = int(((positive == 1) & (predicted == 0)).sum())

    def ratio(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    sensitivity = ratio(tp, tp + fn)
    ppv = ratio(tp, tp + fp)
    f1 = (
        float(2 * sensitivity * ppv / (sensitivity + ppv))
        if sensitivity and ppv and (sensitivity + ppv) > 0
        else None
    )
    degenerate = positive.sum() in (0, positive.size)
    scores = y_score[:, positive_index]
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "sensitivity_recall": sensitivity,
        "specificity": ratio(tn, tn + fp),
        "ppv_precision": ppv,
        "npv": ratio(tn, tn + fn),
        "f1": f1,
        "roc_auc": None if degenerate else float(roc_auc_score(positive, scores)),
        "pr_auc_average_precision": (
            None if degenerate else float(average_precision_score(positive, scores))
        ),
        "positive_prevalence": float(positive.mean()) if positive.size else None,
    }


def pooled_window_metrics(ledger: PredictionLedger) -> dict[str, Any]:
    """All metrics computed over pooled windows.

    **Descriptive only.** Windows within a participant and recording are correlated, so
    these numbers do not carry population uncertainty. The returned dict carries
    ``interpretation="descriptive_only"`` and the report generator prints that label.
    """
    if ledger.frame.empty:
        return {"n_samples": 0, "interpretation": "descriptive_only"}

    y_true, y_pred, y_score = ledger.y_true, ledger.y_pred, ledger.y_score
    n_classes = len(ledger.class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0
    )
    matrices = confusion_matrices(y_true, y_pred, n_classes)
    metrics: dict[str, Any] = {
        "interpretation": "descriptive_only",
        "n_samples": int(len(y_true)),
        "n_subjects": int(len(np.unique(ledger.subjects))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(ledger.class_names)
        },
        "confusion_matrix_raw": matrices["raw"].tolist(),
        "confusion_matrix_row_normalized": matrices["row_normalized"].tolist(),
        "class_names": list(ledger.class_names),
    }
    metrics.update(one_vs_rest_auc(y_true, y_score, ledger.class_names))
    if n_classes == 2:
        metrics["binary"] = binary_metrics(y_true, y_pred, y_score, positive_index=1)
    return metrics


def subject_level_summary(ledger: PredictionLedger) -> dict[str, Any]:
    """Primary aggregation: metrics per participant, then summarised across participants.

    Returns the individual participant results as well as mean / standard deviation /
    range. With five participants no distributional claim is warranted, so the individual
    values are reported and the summary statistics are explicitly descriptive.
    """
    if ledger.frame.empty:
        return {"n_subjects": 0, "per_subject": {}, "interpretation": "primary"}

    per_subject: dict[str, dict[str, Any]] = {}
    for subject in sorted(np.unique(ledger.subjects)):
        subset = ledger.filter(subject_id=subject)
        y_true, y_pred, y_score = subset.y_true, subset.y_pred, subset.y_score
        n_classes = len(ledger.class_names)
        _, _, _, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(n_classes)), zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0
        )
        entry: dict[str, Any] = {
            "n_samples": int(len(y_true)),
            "n_classes_present": int((support > 0).sum()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "class_support": {
                name: int(support[index]) for index, name in enumerate(ledger.class_names)
            },
        }
        entry.update(one_vs_rest_auc(y_true, y_score, ledger.class_names))
        if n_classes == 2:
            entry["binary"] = binary_metrics(y_true, y_pred, y_score, positive_index=1)
        per_subject[str(subject)] = entry

    summary: dict[str, Any] = {
        "interpretation": "primary",
        "n_subjects": len(per_subject),
        "per_subject": per_subject,
        "unit_of_generalization": "participant",
        "note": (
            "With five participants these summary statistics describe the sample; they do "
            "not support a population-level generalization claim."
        ),
    }
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "macro_roc_auc"):
        values = [entry[metric] for entry in per_subject.values() if entry.get(metric) is not None]
        summary[metric] = (
            {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": {subject: entry.get(metric) for subject, entry in per_subject.items()},
                "n": len(values),
            }
            if values
            else None
        )
    return summary


def curve_points(ledger: PredictionLedger, *, max_points: int = 512) -> dict[str, dict[str, Any]]:
    """One-vs-rest ROC and precision-recall curve points per class, for the figures.

    Curves are thinned to at most ``max_points`` evenly spaced vertices so the saved
    artifact stays small; the AUC values reported alongside come from the full arrays.
    """
    out: dict[str, dict[str, Any]] = {}
    if ledger.frame.empty:
        return out
    y_true, y_score = ledger.y_true, ledger.y_score
    for index, name in enumerate(ledger.class_names):
        positive = (y_true == index).astype(np.int64)
        if positive.sum() == 0 or positive.sum() == positive.size:
            out[name] = {"available": False, "reason": "class absent or fully present"}
            continue
        fpr, tpr, _ = roc_curve(positive, y_score[:, index])
        precision, recall, _ = precision_recall_curve(positive, y_score[:, index])

        def thin(array: np.ndarray) -> list[float]:
            if array.size <= max_points:
                return [float(value) for value in array]
            step = np.linspace(0, array.size - 1, max_points).astype(int)
            return [float(value) for value in array[step]]

        out[name] = {
            "available": True,
            "fpr": thin(fpr),
            "tpr": thin(tpr),
            "precision": thin(precision),
            "recall": thin(recall),
            "roc_auc": float(roc_auc_score(positive, y_score[:, index])),
            "average_precision": float(average_precision_score(positive, y_score[:, index])),
            "positive_support": int(positive.sum()),
        }
    return out
