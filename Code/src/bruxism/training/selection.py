"""Prespecified model-selection rules.

Every selection decision this project makes -- which epoch, which hyperparameter trial,
which epoch budget for the final refit -- is made by a rule declared *before* the run, on
inner-fold data only. The rules live here so a reviewer can read them in one place.

The prespecified defaults are:

selection objective
    Validation **macro-F1**, maximised. Macro-F1 rather than accuracy because the
    trigger-constrained protocol produces a strongly imbalanced class distribution in which
    accuracy is dominated by chewing.
tie-break
    Highest macro-F1; then lowest validation loss; then the **earliest** epoch. Earliest
    rather than latest so ties resolve toward the less-overfit model, deterministically.
final epoch budget
    The **median** best epoch across inner folds, rounded up. Median rather than mean
    because a single inner fold that peaks very late should not extend training for all.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from bruxism.utils.logging import get_logger

__all__ = [
    "EpochRecord",
    "SelectionConfig",
    "SelectionResult",
    "TrialResult",
    "select_best_epoch",
    "select_best_trial",
    "select_epoch_budget",
]

logger = get_logger(__name__)

Objective = Literal["macro_f1", "balanced_accuracy", "accuracy", "loss"]


@dataclass(frozen=True)
class EpochRecord:
    """Metrics for one training epoch on one inner fold.

    ``val_*`` fields come from the inner-validation participant. The outer-test participant
    never appears here -- :class:`~bruxism.data.splits.OuterFold` makes it unreachable.
    """

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_balanced_accuracy: float
    val_macro_f1: float
    learning_rate: float = float("nan")

    def objective_value(self, objective: Objective) -> float:
        return {
            "macro_f1": self.val_macro_f1,
            "balanced_accuracy": self.val_balanced_accuracy,
            "accuracy": self.val_accuracy,
            "loss": -self.val_loss,  # negated so every objective is maximised
        }[objective]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionConfig:
    """The prespecified selection rules for a run."""

    objective: Objective = "macro_f1"
    epoch_budget_rule: Literal["median", "mean", "max"] = "median"
    min_epochs: int = 5
    max_epochs: int = 60
    patience: int = 15

    def __post_init__(self) -> None:
        if self.min_epochs < 1 or self.max_epochs < self.min_epochs:
            raise ValueError(
                f"need 1 <= min_epochs <= max_epochs, got {self.min_epochs}/{self.max_epochs}"
            )
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of an epoch selection, with the tie-break trace recorded."""

    best_epoch: int
    best_value: float
    objective: Objective
    n_candidates: int
    tie_broken_by: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_best_epoch(
    history: Sequence[EpochRecord], config: SelectionConfig | None = None
) -> SelectionResult:
    """Pick the best epoch from one inner fold's history.

    Applies the declared tie-break: maximise the objective, then minimise validation loss,
    then take the earliest epoch. The rule is total, so the result never depends on the
    order in which epochs happen to be stored.
    """
    if not history:
        raise ValueError("cannot select an epoch from an empty history")
    active = config or SelectionConfig()

    def sort_key(record: EpochRecord) -> tuple[float, float, int]:
        return (-record.objective_value(active.objective), record.val_loss, record.epoch)

    ordered = sorted(history, key=sort_key)
    best = ordered[0]
    top_value = best.objective_value(active.objective)
    ties = [r for r in history if math.isclose(r.objective_value(active.objective), top_value)]
    if len(ties) == 1:
        reason = "unique_best_objective"
    elif len({r.val_loss for r in ties}) > 1:
        reason = "lowest_val_loss"
    else:
        reason = "earliest_epoch"

    return SelectionResult(
        best_epoch=best.epoch,
        best_value=top_value,
        objective=active.objective,
        n_candidates=len(history),
        tie_broken_by=reason,
    )


def select_epoch_budget(
    inner_results: Sequence[SelectionResult], config: SelectionConfig | None = None
) -> int:
    """Epoch budget for the final refit on all outer-training participants.

    Applies ``config.epoch_budget_rule`` to the inner folds' best epochs and clamps the
    result into ``[min_epochs, max_epochs]``.
    """
    if not inner_results:
        raise ValueError("cannot derive an epoch budget without inner-fold results")
    active = config or SelectionConfig()
    epochs = np.array([result.best_epoch for result in inner_results], dtype=np.float64)
    value = {
        "median": float(np.median(epochs)),
        "mean": float(np.mean(epochs)),
        "max": float(np.max(epochs)),
    }[active.epoch_budget_rule]
    budget = int(min(max(math.ceil(value), active.min_epochs), active.max_epochs))
    logger.info(
        "epoch budget %d from inner best epochs %s (rule=%s)",
        budget,
        epochs.astype(int).tolist(),
        active.epoch_budget_rule,
        extra={"epoch_budget": budget, "inner_best_epochs": epochs.astype(int).tolist()},
    )
    return budget


@dataclass
class TrialResult:
    """One hyperparameter trial, scored across all inner folds of one outer fold."""

    trial_id: str
    hyperparameters: dict[str, Any]
    inner_selections: list[SelectionResult] = field(default_factory=list)
    inner_fold_subjects: list[str] = field(default_factory=list)
    failed: bool = False
    failure_reason: str = ""

    @property
    def mean_objective(self) -> float:
        """Mean of the per-inner-fold best objective values. NaN if the trial failed."""
        if self.failed or not self.inner_selections:
            return float("nan")
        return float(np.mean([result.best_value for result in self.inner_selections]))

    @property
    def std_objective(self) -> float:
        if self.failed or len(self.inner_selections) < 2:
            return 0.0
        return float(np.std([result.best_value for result in self.inner_selections]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "hyperparameters": self.hyperparameters,
            "mean_objective": self.mean_objective,
            "std_objective": self.std_objective,
            "inner_selections": [result.to_dict() for result in self.inner_selections],
            "inner_fold_subjects": self.inner_fold_subjects,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
        }


def select_best_trial(trials: Sequence[TrialResult]) -> TrialResult:
    """Pick the winning hyperparameter trial for one outer fold.

    Maximises the mean inner objective; ties break toward the **lower** across-fold
    standard deviation (more stable configuration), then toward the lexicographically
    smallest ``trial_id`` so the result is deterministic.

    Raises
    ------
    ValueError
        If every trial failed. A silent fallback to an arbitrary configuration would hide
        a broken search space.
    """
    usable = [trial for trial in trials if not trial.failed and trial.inner_selections]
    if not usable:
        reasons = {trial.trial_id: trial.failure_reason for trial in trials}
        raise ValueError(f"every hyperparameter trial failed: {reasons}")
    if len(usable) < len(trials):
        logger.warning(
            "%d of %d trials failed and were excluded from selection",
            len(trials) - len(usable),
            len(trials),
        )
    return sorted(
        usable, key=lambda trial: (-trial.mean_objective, trial.std_objective, trial.trial_id)
    )[0]
