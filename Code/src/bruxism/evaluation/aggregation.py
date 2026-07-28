"""Aggregation across conditions, seeds and participants.

Everything here reads a prediction ledger and nothing else. In particular:

* multi-seed handling is **prespecified**, not chosen after seeing results:
  :func:`summarise_ledger` reports per-seed metrics *and* the across-seed summary, and
  never selects a best seed;
* the modality contrast (fusion vs EMG-only vs audio-only) is computed per participant and
  reported as paired differences, because a pooled difference on this dataset is dominated
  by chewing;
* the chewing contrast pairs the five-class result with its no-chewing twin, which is the
  analysis that decides whether the audio contribution survives once eating is removed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from bruxism.data.labels import get_task
from bruxism.evaluation.metrics import (
    PredictionLedger,
    curve_points,
    pooled_window_metrics,
    subject_level_summary,
)
from bruxism.utils.logging import get_logger

__all__ = [
    "MULTISEED_RULE",
    "chewing_contrast",
    "condition_table",
    "modality_contrast",
    "summarise_ledger",
]

logger = get_logger(__name__)

#: Declared in advance: seeds are summarised, never selected between.
MULTISEED_RULE = (
    "Metrics are computed independently per seed, then summarised across seeds as "
    "mean/std/min/max. Probabilities are NOT averaged across seeds, and no seed is "
    "selected as best."
)

_GROUP_KEYS = ("task_id", "model_id", "modality", "seed")


def _condition_id(row: pd.Series | dict[str, Any]) -> str:
    return f"{row['task_id']}::{row['model_id']}::{row['modality']}::seed{int(row['seed'])}"


def summarise_ledger(predictions: pd.DataFrame, *, include_curves: bool = True) -> dict[str, Any]:
    """Full metric summary of a prediction ledger.

    Returns a dict with ``conditions`` (one entry per task/model/modality/seed),
    ``across_seeds`` (per task/model/modality), and the contrast analyses.
    """
    if predictions.empty:
        return {"conditions": {}, "across_seeds": {}, "multiseed_rule": MULTISEED_RULE}

    conditions: dict[str, Any] = {}
    for keys, group in predictions.groupby(list(_GROUP_KEYS), observed=True):
        task_id, model_id, modality, seed = (str(k) for k in keys)  # type: ignore[misc]
        task = get_task(task_id)
        ledger = PredictionLedger(frame=group.reset_index(drop=True), class_names=task.class_names)
        ledger.assert_exactly_once()
        entry: dict[str, Any] = {
            "task_id": task_id,
            "model_id": model_id,
            "modality": modality,
            "seed": int(float(str(seed))),
            "class_names": list(task.class_names),
            "task_description": task.description,
            "task_notes": task.notes,
            "is_primary_endpoint": task.primary,
            "safe_for_inference": bool(group.get("safe_for_inference", pd.Series([True])).all()),
            "subject_level": subject_level_summary(ledger),
            "pooled_windows": pooled_window_metrics(ledger),
        }
        if include_curves:
            entry["curves"] = curve_points(ledger)
        conditions[_condition_id(group.iloc[0])] = entry

    return {
        "multiseed_rule": MULTISEED_RULE,
        "conditions": conditions,
        "across_seeds": _across_seeds(conditions),
        "modality_contrast": modality_contrast(predictions),
        "chewing_contrast": chewing_contrast(predictions),
        "sample_counts": _sample_counts(predictions),
    }


def _across_seeds(conditions: dict[str, Any]) -> dict[str, Any]:
    """Summarise repeated seeds per (task, model, modality) without selecting one."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for entry in conditions.values():
        key = (entry["task_id"], entry["model_id"], entry["modality"])
        grouped.setdefault(key, []).append(entry)

    summary: dict[str, Any] = {}
    for (task_id, model_id, modality), entries in sorted(grouped.items()):
        block: dict[str, Any] = {
            "task_id": task_id,
            "model_id": model_id,
            "modality": modality,
            "n_seeds": len(entries),
            "seeds": sorted(entry["seed"] for entry in entries),
        }
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "macro_roc_auc"):
            values = [
                entry["subject_level"].get(metric, {}).get("mean")
                for entry in entries
                if isinstance(entry["subject_level"].get(metric), dict)
            ]
            values = [value for value in values if value is not None]
            block[f"subject_mean_{metric}"] = (
                {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_seeds": len(values),
                }
                if values
                else None
            )
        summary[f"{task_id}::{model_id}::{modality}"] = block
    return summary


def _per_subject_metric(predictions: pd.DataFrame, *, metric: str = "macro_f1") -> pd.DataFrame:
    """Long table of one metric per (condition, seed, participant)."""
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(list(_GROUP_KEYS), observed=True):
        task_id, model_id, modality, seed = (str(k) for k in keys)  # type: ignore[misc]
        task = get_task(task_id)
        ledger = PredictionLedger(frame=group.reset_index(drop=True), class_names=task.class_names)
        summary = subject_level_summary(ledger)
        for subject, entry in summary.get("per_subject", {}).items():
            rows.append(
                {
                    "task_id": task_id,
                    "model_id": model_id,
                    "modality": modality,
                    "seed": int(float(str(seed))),
                    "subject_id": subject,
                    "metric": metric,
                    "value": entry.get(metric),
                    "n_samples": entry.get("n_samples"),
                }
            )
    return pd.DataFrame(rows)


def modality_contrast(
    predictions: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("macro_f1", "balanced_accuracy", "accuracy"),
    reference: str = "emg_only",
    comparison: str = "fusion",
) -> dict[str, Any]:
    """Per-participant paired differences between modality conditions.

    Reports the participant-wise difference ``comparison - reference`` for each task and
    model, plus the individual values. With five participants no inferential test is
    performed: the paired differences and their range are the evidence, and the returned
    dict says so.
    """
    out: dict[str, Any] = {
        "reference": reference,
        "comparison": comparison,
        "interpretation": (
            "Paired per-participant differences. With five participants these are "
            "descriptive; no significance test is reported because the unit count is far "
            "too small for one to be meaningful."
        ),
        "contrasts": {},
    }
    for metric in metrics:
        table = _per_subject_metric(predictions, metric=metric)
        if table.empty:
            continue
        wide = table.pivot_table(
            index=["task_id", "model_id", "seed", "subject_id"],
            columns="modality",
            values="value",
        ).reset_index()
        if reference not in wide.columns or comparison not in wide.columns:
            continue
        wide["difference"] = wide[comparison] - wide[reference]
        for group_key, group in wide.groupby(["task_id", "model_id", "seed"], observed=True):
            task_id, model_id, seed = (str(part) for part in group_key)  # type: ignore[misc]
            seed_number = int(float(seed))
            key = f"{task_id}::{model_id}::seed{seed_number}::{metric}"
            differences = group["difference"].dropna()
            out["contrasts"][key] = {
                "task_id": task_id,
                "model_id": model_id,
                "seed": seed_number,
                "metric": metric,
                "per_subject": {
                    str(row.subject_id): {
                        reference: _nan_to_none(getattr(row, reference)),
                        comparison: _nan_to_none(getattr(row, comparison)),
                        "difference": _nan_to_none(row.difference),
                    }
                    for row in group.itertuples()
                },
                "mean_difference": float(differences.mean()) if len(differences) else None,
                "min_difference": float(differences.min()) if len(differences) else None,
                "max_difference": float(differences.max()) if len(differences) else None,
                "n_subjects_improved": int((differences > 0).sum()),
                "n_subjects": int(len(differences)),
            }
    return out


def chewing_contrast(
    predictions: pd.DataFrame,
    *,
    with_chewing: str = "five_class",
    without_chewing: str = "no_chewing_four_class",
    metric: str = "macro_f1",
) -> dict[str, Any]:
    """Does the audio contribution survive once chewing is removed?

    Computes the fusion-minus-EMG-only difference on the with-chewing task and on the
    no-chewing task, and reports both. A benefit that exists only in the first case means
    the microphone is mainly detecting eating, and the manuscript must say so.
    """
    result: dict[str, Any] = {
        "metric": metric,
        "with_chewing_task": with_chewing,
        "without_chewing_task": without_chewing,
        "interpretation": (
            "If the fusion-minus-EMG-only gain is present with chewing but absent without "
            "it, the audio branch is primarily separating eating rather than contributing "
            "to the clench / instructed-grinding distinction."
        ),
        "available": False,
    }
    table = _per_subject_metric(predictions, metric=metric)
    if table.empty:
        return result
    present = set(table["task_id"])
    if with_chewing not in present or without_chewing not in present:
        result["reason"] = (
            f"needs both {with_chewing!r} and {without_chewing!r} in the ledger; "
            f"found {sorted(present)}"
        )
        return result

    contrasts = modality_contrast(predictions, metrics=(metric,))
    picked: dict[str, Any] = {}
    for key, entry in contrasts["contrasts"].items():
        if entry["task_id"] in (with_chewing, without_chewing):
            picked[key] = entry
    result["available"] = bool(picked)
    result["contrasts"] = picked
    return result


def _sample_counts(predictions: pd.DataFrame) -> dict[str, Any]:
    """Task-specific sample counts by participant and source condition."""
    counts: dict[str, Any] = {}
    for task_id, group in predictions.groupby("task_id", observed=True):
        one_seed = group[group["seed"] == group["seed"].min()]
        one_model = one_seed[one_seed["model_id"] == sorted(one_seed["model_id"].unique())[0]]
        one = one_model[one_model["modality"] == sorted(one_model["modality"].unique())[0]]
        counts[str(task_id)] = {
            "by_subject_and_class": (
                one.groupby(["subject_id", "true_class"], observed=True)
                .size()
                .reset_index(name="n")
                .to_dict("records")
            ),
            "by_condition": (
                one.groupby(["condition", "true_class"], observed=True)
                .size()
                .reset_index(name="n")
                .to_dict("records")
                if "condition" in one.columns
                else []
            ),
            "total": int(len(one)),
        }
    return counts


def condition_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Tidy comparison table across conditions, for the LaTeX generator."""
    rows: list[dict[str, Any]] = []
    for condition_id, entry in metrics.get("conditions", {}).items():
        subject = entry.get("subject_level", {})
        pooled = entry.get("pooled_windows", {})
        row: dict[str, Any] = {
            "condition_id": condition_id,
            "task_id": entry.get("task_id"),
            "model_id": entry.get("model_id"),
            "modality": entry.get("modality"),
            "seed": entry.get("seed"),
            "is_primary_endpoint": entry.get("is_primary_endpoint"),
            "safe_for_inference": entry.get("safe_for_inference"),
            "n_windows_descriptive": pooled.get("n_samples"),
        }
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "macro_roc_auc"):
            block = subject.get(metric)
            if isinstance(block, dict):
                row[f"{metric}_mean"] = block.get("mean")
                row[f"{metric}_std"] = block.get("std")
                row[f"{metric}_min"] = block.get("min")
                row[f"{metric}_max"] = block.get("max")
        row["macro_average_precision_pooled"] = pooled.get("macro_average_precision")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["task_id", "model_id", "modality", "seed"], ignore_index=True
    )


def _nan_to_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(numeric) else numeric
