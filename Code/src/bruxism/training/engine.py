"""The nested leave-one-subject-out training engine.

Protocol executed per outer fold:

1. Seal one participant as the outer test set (:class:`~bruxism.data.splits.OuterFold`
   refuses to hand out its ids except for ``purpose="final_evaluation"``).
2. Run participant-grouped inner LOSO over the remaining participants -- four folds when
   four training participants remain, never five.
3. Fit normalisation, class weights, augmentation minority sets and every hyperparameter on
   inner-training participants only.
4. Select the best epoch per inner fold by the prespecified objective
   (:mod:`bruxism.training.selection`).
5. Derive the final epoch budget from the inner best epochs by the prespecified rule.
6. Refit on all outer-training participants for exactly that many epochs.
7. Evaluate the held-out participant **once** and append to the prediction ledger.

Checkpoints are deep-copied at selection time and written atomically, so a later optimiser
step cannot mutate the "best" weights -- the prototype's ``state_dict().copy()`` was a
shallow copy whose tensors kept changing.
"""

from __future__ import annotations

import copy
import itertools
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from bruxism.config import ExperimentConfig
from bruxism.data.dataset import RecordingCache, WindowDataset, collate_windows
from bruxism.data.labels import ClassificationTask
from bruxism.data.segments import WindowIndex
from bruxism.data.splits import NestedLOSOSplitter, OuterFold
from bruxism.evaluation.metrics import LEDGER_COLUMNS
from bruxism.models import BruxismModel
from bruxism.models.baselines import build_neural_model
from bruxism.preprocessing.augmentation import Augmenter
from bruxism.preprocessing.calibration import CalibrationBlock
from bruxism.preprocessing.normalization import Normalizer
from bruxism.training.losses import build_loss, compute_class_weights
from bruxism.training.selection import (
    EpochRecord,
    TrialResult,
    select_best_epoch,
    select_best_trial,
    select_epoch_budget,
)
from bruxism.utils import progress
from bruxism.utils.io import atomic_write_bytes, file_sha256
from bruxism.utils.logging import get_logger
from bruxism.utils.reproducibility import SeedBundle, seed_everything, worker_init_fn

__all__ = ["FoldOutcome", "NestedLOSOTrainer", "resolve_device"]

logger = get_logger(__name__)


def resolve_device(preference: str) -> torch.device:
    """Resolve ``"auto" | "cpu" | "cuda"`` into a concrete device, failing loudly."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but no CUDA device is available")
        return torch.device("cuda")
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unknown device preference {preference!r}")


def _describe_hyperparameters(overrides: Mapping[str, Any]) -> str:
    """Compact rendering of one trial's hyperparameters, for progress and log lines."""
    if not overrides:
        return "config defaults"
    return " ".join(f"{key}={value}" for key, value in sorted(overrides.items()))


@dataclass
class FoldOutcome:
    """Everything one outer fold produced, saved before the next fold starts."""

    outer_fold: int
    test_subject: str
    seed: int
    epoch_budget: int
    selected_hyperparameters: dict[str, Any]
    inner_trials: list[dict[str, Any]]
    training_history: list[dict[str, Any]]
    #: What the ``val_*`` columns of ``training_history`` were measured on. For the final
    #: refit this is ``"refit_training_fit"`` -- they are fit diagnostics, not validation.
    history_scope: str
    normalizer: dict[str, Any]
    class_weights: list[float]
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    parameter_counts: dict[str, int]
    n_train: int
    n_test: int
    train_seconds: float
    augmentation_stats: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class NestedLOSOTrainer:
    """Runs the nested LOSO protocol for one (task, model, modality, seed) condition."""

    def __init__(
        self,
        config: ExperimentConfig,
        window_index: WindowIndex,
        cache: RecordingCache,
        task: ClassificationTask,
        *,
        run_dir: Path,
        source_commit: str | None,
        manifest_hash: str,
        model_id: str | None = None,
        modality: str | None = None,
        calibration_block: CalibrationBlock | None = None,
    ):
        self.config = config
        self.window_index = window_index
        self.cache = cache
        self.task = task
        self.run_dir = Path(run_dir)
        self.source_commit = source_commit
        self.manifest_hash = manifest_hash
        self.model_id = model_id or config.model_id
        self.modality = modality or config.modality
        self.device = resolve_device(config.training.device)
        self.selection = config.selection
        self.calibration_block = calibration_block

        if config.normalization.scope == "per_participant":
            if calibration_block is None:
                raise ValueError(
                    "normalization.scope='per_participant' requires a calibration block; "
                    "bruxism.runner builds one and passes it in"
                )
            logger.warning(
                "TRANSDUCTIVE PROTOCOL: per-participant normalisation uses each "
                "participant's own unlabelled calibration block (%s), including the "
                "held-out participant. Results from this run require a fitting session "
                "and are NOT comparable to the strict, training-set-only arm.",
                calibration_block.source,
            )

        if not window_index.safe_for_inference:
            logger.warning(
                "training on a DIAGNOSTIC-ONLY segmentation policy (%s); results must be "
                "labelled unsafe for inference",
                window_index.config.policy,
            )

    # ------------------------------------------------------------------ helpers ---

    def _make_loader(
        self, dataset: WindowDataset, *, shuffle: bool, seed: int, batch_size: int
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(SeedBundle.from_base(seed).dataloader_seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.training.num_workers,
            collate_fn=collate_windows,
            worker_init_fn=worker_init_fn if self.config.training.num_workers else None,
            generator=generator,
            drop_last=shuffle and len(dataset) > batch_size,
            pin_memory=self.device.type == "cuda",
        )

    def _fit_normalizer(
        self,
        sample_ids: Sequence[str],
        subjects: Sequence[str],
        *,
        calibrate_subjects: Sequence[str] = (),
    ) -> Normalizer:
        """Fit normalisation statistics on the given (training-only) sample ids.

        Parameters
        ----------
        calibrate_subjects
            Participants to additionally calibrate from their own calibration block, when
            ``normalization.scope`` is ``"per_participant"``. This is where the held-out
            participant's *unlabelled* signal legitimately enters; the block itself was
            withheld from every split by the splitter, and the run bundle records which
            windows produced each participant's statistics.
        """
        wanted = set(sample_ids)
        windows = [w for w in self.window_index.windows if w.sample_id in wanted]
        emg, mic = self.cache.training_statistics_source(windows)
        normalizer = Normalizer(self.config.normalization).fit(emg, mic, subjects=tuple(subjects))

        if self.config.normalization.scope != "per_participant":
            return normalizer
        block = self.calibration_block
        if block is None:
            raise ValueError(
                "normalization.scope='per_participant' but no calibration block was built; "
                "the trainer must be constructed with one"
            )
        for subject in sorted({*subjects, *calibrate_subjects}):
            ids = set(block.for_subject(subject))
            if not ids:
                logger.warning(
                    "participant %s has no calibration block; falling back to the pooled "
                    "training statistics for them",
                    subject,
                )
                continue
            block_windows = [w for w in self.window_index.windows if w.sample_id in ids]
            block_emg, block_mic = self.cache.training_statistics_source(
                block_windows, max_windows=None
            )
            normalizer.calibrate(subject, block_emg, block_mic, sample_ids=tuple(sorted(ids)))
        return normalizer

    def _build_model(self) -> BruxismModel:
        model = build_neural_model(
            self.model_id,
            num_classes=self.task.num_classes,
            modality=self.modality,  # type: ignore[arg-type]
            emg_channels=self.cache.n_emg_channels,
            window_samples=self.config.data.window_samples(),
            overrides=self.config.model_overrides,
        )
        return cast("BruxismModel", model.to(self.device))

    def _build_optimizer(self, model: BruxismModel, training: Any) -> torch.optim.Optimizer:
        params = model.parameters()
        if training.optimizer == "adam":
            return torch.optim.Adam(
                params, lr=training.learning_rate, weight_decay=training.weight_decay
            )
        if training.optimizer == "adamw":
            return torch.optim.AdamW(
                params, lr=training.learning_rate, weight_decay=training.weight_decay
            )
        if training.optimizer == "sgd":
            return torch.optim.SGD(
                params,
                lr=training.learning_rate,
                weight_decay=training.weight_decay,
                momentum=0.9,
            )
        raise ValueError(f"unknown optimizer {training.optimizer!r}")

    def _make_datasets(
        self,
        train_ids: Sequence[str],
        eval_ids: Sequence[str],
        *,
        eval_stage: str,
        normalizer: Normalizer,
        seed: int,
        augment: bool,
    ) -> tuple[WindowDataset, WindowDataset]:
        train_probe = WindowDataset(
            self.window_index,
            train_ids,
            self.task,
            self.cache,
            stage="train",
            normalizer=normalizer,
            modality=self.modality,  # type: ignore[arg-type]
        )
        augmenter = None
        if augment and self.config.augmentation.enabled:
            minority = Augmenter.minority_labels_from_counts(
                train_probe.label_counts(),
                threshold=self.config.augmentation.minority_threshold,
            )
            augmenter = Augmenter(self.config.augmentation, run_seed=seed, minority_labels=minority)
        train_dataset = WindowDataset(
            self.window_index,
            train_ids,
            self.task,
            self.cache,
            stage="train",
            normalizer=normalizer,
            augmenter=augmenter,
            modality=self.modality,  # type: ignore[arg-type]
        )
        eval_dataset = WindowDataset(
            self.window_index,
            eval_ids,
            self.task,
            self.cache,
            stage=eval_stage,  # type: ignore[arg-type]
            normalizer=normalizer,
            modality=self.modality,  # type: ignore[arg-type]
        )
        return train_dataset, eval_dataset

    # ------------------------------------------------------------------- epochs ---

    def _train_one_epoch(
        self,
        model: BruxismModel,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        grad_clip: float | None,
    ) -> float:
        model.train()
        total, seen = 0.0, 0
        for emg, mic, labels, _ in loader:
            emg = emg.to(self.device, non_blocking=True)
            mic = mic.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(emg, mic)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total += float(loss.item()) * labels.size(0)
            seen += int(labels.size(0))
        return total / max(seen, 1)

    @torch.no_grad()
    def _evaluate(
        self, model: BruxismModel, loader: DataLoader, criterion: nn.Module
    ) -> dict[str, Any]:
        """Forward pass returning loss, labels, predictions, probabilities and indices."""
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            precision_recall_fscore_support,
        )

        model.eval()
        losses, seen = 0.0, 0
        all_labels: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []
        all_indices: list[np.ndarray] = []
        for emg, mic, labels, indices in loader:
            emg = emg.to(self.device, non_blocking=True)
            mic = mic.to(self.device, non_blocking=True)
            labels_device = labels.to(self.device, non_blocking=True)
            logits = model(emg, mic)
            losses += float(criterion(logits, labels_device).item()) * labels.size(0)
            seen += int(labels.size(0))
            all_probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            all_labels.append(labels.numpy())
            all_indices.append(indices.numpy())

        if not all_labels:
            raise RuntimeError("evaluation loader produced no batches")
        y_true = np.concatenate(all_labels)
        probs = np.concatenate(all_probs)
        y_pred = probs.argmax(axis=1)
        _, _, macro_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(self.task.num_classes)),
            average="macro",
            zero_division=0,
        )
        return {
            "loss": losses / max(seen, 1),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(macro_f1),
            "y_true": y_true,
            "y_pred": y_pred,
            "probabilities": probs,
            "indices": np.concatenate(all_indices),
        }

    def _train_loop(
        self,
        train_dataset: WindowDataset,
        eval_dataset: WindowDataset,
        training: Any,
        *,
        seed: int,
        max_epochs: int,
        capture_best: bool,
        disable_early_stopping: bool = False,
        label: str = "epochs",
        bar_label: str = "epochs",
        eval_name: str = "val",
    ) -> tuple[list[EpochRecord], dict[str, torch.Tensor] | None, BruxismModel]:
        """Train, recording per-epoch validation metrics.

        When ``capture_best`` is set, the state dict at the best epoch is **deep-copied**
        so subsequent optimiser steps cannot mutate it.

        ``label`` names this fit in log lines and ``bar_label`` is its shorter form for a
        drawn bar, which already sits under a bar naming the fold. ``eval_name`` names what
        the per-epoch evaluation was measured on -- ``"val"`` for an inner fold, ``"fit"``
        for the final refit, whose numbers come from the training data and are a fit
        diagnostic rather than validation.
        """
        seed_everything(seed, deterministic=training.deterministic)
        model = self._build_model()
        optimizer = self._build_optimizer(model, training)

        weights = compute_class_weights(
            train_dataset.labels,
            self.task.num_classes,
            scheme=training.class_weight_scheme,
        )
        criterion = build_loss(
            training.loss,
            class_weights=weights,
            gamma=training.focal_gamma,
            label_smoothing=training.label_smoothing,
            device=self.device,
        )
        train_loader = self._make_loader(
            train_dataset, shuffle=True, seed=seed, batch_size=training.batch_size
        )
        eval_loader = self._make_loader(
            eval_dataset, shuffle=False, seed=seed, batch_size=training.batch_size
        )

        history: list[EpochRecord] = []
        best_state: dict[str, torch.Tensor] | None = None
        best_value = -np.inf
        stale = 0

        objective = self.selection.objective
        with progress.task(
            label, total=max_epochs, unit="epoch", bar_description=f"    {bar_label}"
        ) as epochs:
            for epoch in range(1, max_epochs + 1):
                train_dataset.set_epoch(epoch)
                train_loss = self._train_one_epoch(
                    model, train_loader, criterion, optimizer, grad_clip=training.grad_clip_norm
                )
                evaluation = self._evaluate(model, eval_loader, criterion)
                record = EpochRecord(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=evaluation["loss"],
                    val_accuracy=evaluation["accuracy"],
                    val_balanced_accuracy=evaluation["balanced_accuracy"],
                    val_macro_f1=evaluation["macro_f1"],
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                )
                history.append(record)

                value = record.objective_value(objective)
                epochs.update(
                    1,
                    **{
                        "loss": round(train_loss, 4),
                        f"{eval_name}_loss": round(float(evaluation["loss"]), 4),
                        f"{eval_name}_{objective}": round(float(value), 4),
                        "best": round(float(max(best_value, value)), 4),
                    },
                )
                logger.debug(
                    "%s epoch %d/%d: train_loss=%.4f %s_loss=%.4f %s_%s=%.4f",
                    label,
                    epoch,
                    max_epochs,
                    train_loss,
                    eval_name,
                    evaluation["loss"],
                    eval_name,
                    objective,
                    value,
                    extra={
                        "epoch": epoch,
                        "max_epochs": max_epochs,
                        "train_loss": train_loss,
                        "eval_scope": eval_name,
                        "eval_loss": float(evaluation["loss"]),
                        "objective": objective,
                        "objective_value": float(value),
                    },
                )

                if value > best_value:
                    best_value = value
                    stale = 0
                    if capture_best:
                        # Deep copy: a shallow state_dict() copy shares tensor storage with
                        # the live model and would keep changing as training continues.
                        best_state = copy.deepcopy(
                            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        )
                else:
                    stale += 1
                if (
                    not disable_early_stopping
                    and epoch >= self.selection.min_epochs
                    and stale >= self.selection.patience
                ):
                    logger.debug(
                        "early stop at epoch %d (patience %d)", epoch, self.selection.patience
                    )
                    break
        return history, best_state, model

    # ------------------------------------------------------------ hyperparameters ---

    def _trial_grid(self) -> list[dict[str, Any]]:
        """Cartesian product of the declared search space; empty space -> one trial."""
        space = self.config.training.search_space
        if not space:
            return [{}]
        keys = sorted(space)
        return [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]

    def _run_inner_search(
        self, fold: OuterFold, seed: int, *, label: str = ""
    ) -> tuple[TrialResult, list[dict]]:
        """Score every hyperparameter trial on every inner fold of this outer fold."""
        grid = self._trial_grid()
        n_inner = fold.n_inner_folds
        objective = self.selection.objective
        logger.info(
            "%sinner search: %d trial(s) x %d inner fold(s) = %d model fit(s), "
            "up to %d epochs each",
            f"{label} " if label else "",
            len(grid),
            n_inner,
            len(grid) * n_inner,
            self.selection.max_epochs,
            extra={"n_trials": len(grid), "n_inner_folds": n_inner},
        )

        trials: list[TrialResult] = []
        with progress.task(
            f"{label} inner search".strip(),
            total=len(grid) * n_inner,
            unit="fit",
            bar_description="  inner search",
        ) as search:
            for trial_index, overrides in enumerate(grid):
                trial_id = f"trial{trial_index:02d}"
                settings = _describe_hyperparameters(overrides)
                training = self.config.training.replace(**overrides)
                result = TrialResult(trial_id=trial_id, hyperparameters=dict(overrides))
                trial_started = time.perf_counter()
                try:
                    for inner in fold.inner_folds:
                        fit_label = (
                            f"{label} {trial_id} inner {inner.fold_index + 1}/{n_inner} "
                            f"val={inner.val_subject}"
                        ).strip()
                        search.set_description(
                            fit_label,
                            bar_description=(
                                f"  inner search {trial_id} fold {inner.fold_index + 1}/{n_inner} "
                                f"val={inner.val_subject}"
                            ),
                        )
                        fit_started = time.perf_counter()

                        normalizer = self._fit_normalizer(
                            inner.train_sample_ids,
                            inner.train_subjects,
                            # The inner validation participant is calibrated so selection
                            # sees the same protocol the final evaluation will. The outer
                            # test participant is NOT: the seal covers their signal too.
                            calibrate_subjects=[inner.val_subject],
                        )
                        normalizer.assert_not_fitted_on([fold.test_subject, inner.val_subject])
                        normalizer.assert_calibration_disjoint_from(
                            [*inner.train_sample_ids, *inner.val_sample_ids]
                        )
                        train_dataset, val_dataset = self._make_datasets(
                            inner.train_sample_ids,
                            inner.val_sample_ids,
                            eval_stage="val",
                            normalizer=normalizer,
                            seed=seed,
                            augment=True,
                        )
                        if len(train_dataset) == 0 or len(val_dataset) == 0:
                            raise ValueError(
                                f"inner fold {inner.fold_index} (val={inner.val_subject}) has "
                                f"{len(train_dataset)} train / {len(val_dataset)} val examples "
                                f"for task {self.task.task_id}"
                            )
                        history, _, _ = self._train_loop(
                            train_dataset,
                            val_dataset,
                            training,
                            seed=seed,
                            max_epochs=self.selection.max_epochs,
                            capture_best=False,
                            label=f"{fit_label} epochs",
                            bar_label="epochs",
                            eval_name="val",
                        )
                        selected = select_best_epoch(history, self.selection)
                        result.inner_selections.append(selected)
                        result.inner_fold_subjects.append(inner.val_subject)
                        search.update(1)
                        logger.info(
                            "%s [%s]: best val_%s=%.4f at epoch %d of %d run in %s",
                            fit_label,
                            settings,
                            objective,
                            selected.best_value,
                            selected.best_epoch,
                            len(history),
                            progress.format_duration(time.perf_counter() - fit_started),
                            extra={
                                "trial_id": trial_id,
                                "val_subject": inner.val_subject,
                                "best_epoch": selected.best_epoch,
                                "best_value": selected.best_value,
                                "n_epochs_run": len(history),
                            },
                        )
                except Exception as exc:  # noqa: BLE001 - failure is recorded, never hidden
                    result.failed = True
                    result.failure_reason = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "trial %s failed on outer fold %s: %s", trial_id, fold.test_subject, exc
                    )
                    # Keep the bar truthful: charge the fits this trial will never run.
                    search.update(n_inner - len(result.inner_selections))
                trials.append(result)
                if not result.failed:
                    logger.info(
                        "%s%s [%s] done: mean val_%s=%.4f +/- %.4f over %d inner fold(s) in %s",
                        f"{label} " if label else "",
                        trial_id,
                        settings,
                        objective,
                        result.mean_objective,
                        result.std_objective,
                        len(result.inner_selections),
                        progress.format_duration(time.perf_counter() - trial_started),
                        extra={
                            "trial_id": trial_id,
                            "mean_objective": result.mean_objective,
                            "std_objective": result.std_objective,
                        },
                    )

        best = select_best_trial(trials)
        logger.info(
            "%sselected %s [%s] with mean val_%s=%.4f",
            f"{label} " if label else "",
            best.trial_id,
            _describe_hyperparameters(best.hyperparameters),
            objective,
            best.mean_objective,
        )
        return best, [trial.to_dict() for trial in trials]

    # --------------------------------------------------------------------- folds ---

    def run_fold(
        self, fold: OuterFold, seed: int, *, progress_label: str = ""
    ) -> tuple[FoldOutcome, pd.DataFrame]:
        """Execute the full protocol for one outer fold and one seed.

        ``progress_label`` prefixes this fold's progress and log lines with its position in
        the wider run (``fold 3/15``); it is display only.
        """
        started = time.perf_counter()
        label = f"{progress_label} {fold.test_subject} seed{seed}".strip()
        logger.info(
            "%s: outer fold %d, held-out participant %s (train=%s), seed %d, "
            "task=%s model=%s modality=%s device=%s",
            label,
            fold.fold_index,
            fold.test_subject,
            ",".join(fold.train_subjects),
            seed,
            self.task.task_id,
            self.model_id,
            self.modality,
            self.device.type,
            extra={"outer_fold": fold.fold_index, "test_subject": fold.test_subject, "seed": seed},
        )

        best_trial, all_trials = self._run_inner_search(fold, seed, label=label)
        epoch_budget = select_epoch_budget(best_trial.inner_selections, self.selection)
        training = self.config.training.replace(**best_trial.hyperparameters)

        # Final refit on ALL outer-training participants for the fixed epoch budget.
        normalizer = self._fit_normalizer(
            fold.train_sample_ids,
            fold.train_subjects,
            calibrate_subjects=[fold.test_subject],
        )
        # Unchanged meaning: the held-out participant contributed no LABELLED training
        # window to the pooled statistics. Their unlabelled calibration block may have
        # contributed to their own per-participant statistics, which is recorded
        # separately in normalizer.calibrated_on and disclosed in the run bundle.
        normalizer.assert_not_fitted_on([fold.test_subject])

        test_ids = fold.release_test_ids(purpose="final_evaluation")
        normalizer.assert_calibration_disjoint_from([*fold.train_sample_ids, *test_ids])
        train_dataset, test_dataset = self._make_datasets(
            fold.train_sample_ids,
            test_ids,
            eval_stage="test",
            normalizer=normalizer,
            seed=seed,
            augment=True,
        )
        if len(test_dataset) == 0:
            raise ValueError(
                f"held-out participant {fold.test_subject} has no examples for task "
                f"{self.task.task_id}; it cannot be evaluated"
            )

        # The refit runs for exactly `epoch_budget` epochs with no early stopping and no
        # validation set: the epoch count was already decided on inner folds. The per-epoch
        # numbers recorded here are measured on the TRAINING data, so they are a fit
        # diagnostic, not validation, and are labelled `history_scope` accordingly. The
        # held-out participant is not touched until the single evaluation below.
        refit_started = time.perf_counter()
        logger.info(
            "%s: refitting on %d windows from %d participant(s) for exactly %d epoch(s)",
            label,
            len(train_dataset),
            len(fold.train_subjects),
            epoch_budget,
            extra={"epoch_budget": epoch_budget, "n_train": len(train_dataset)},
        )
        history, _, model = self._train_loop(
            train_dataset,
            train_dataset,
            training,
            seed=seed,
            max_epochs=epoch_budget,
            capture_best=False,
            disable_early_stopping=True,
            label=f"{label} refit epochs".strip(),
            bar_label="refit epochs",
            eval_name="fit",
        )
        logger.info(
            "%s: refit done in %s (final train loss %.4f)",
            label,
            progress.format_duration(time.perf_counter() - refit_started),
            history[-1].train_loss,
        )

        class_weights = compute_class_weights(
            train_dataset.labels, self.task.num_classes, scheme=training.class_weight_scheme
        )
        criterion = build_loss(
            training.loss,
            class_weights=class_weights,
            gamma=training.focal_gamma,
            label_smoothing=training.label_smoothing,
            device=self.device,
        )

        checkpoint_path: Path | None = None
        checkpoint_sha: str | None = None
        if self.config.output.save_checkpoints:
            checkpoint_path = (
                self.run_dir
                / "checkpoints"
                / f"{self.task.task_id}_{self.model_id}_{self.modality}"
                f"_fold{fold.fold_index}_seed{seed}.pt"
            )
            payload = {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "architecture": model.architecture_record(self.config.data.sampling_rate_hz),
                "normalizer": normalizer.to_dict(),
                "class_weights": class_weights.tolist(),
                "epoch_budget": epoch_budget,
                "hyperparameters": best_trial.hyperparameters,
                "task_id": self.task.task_id,
                "class_names": list(self.task.class_names),
                "config_hash": self.config.config_hash,
                "manifest_hash": self.manifest_hash,
                "seed": seed,
            }
            import io

            buffer = io.BytesIO()
            torch.save(payload, buffer)
            atomic_write_bytes(checkpoint_path, buffer.getvalue())
            checkpoint_sha = file_sha256(checkpoint_path)

        # ---- the single, final evaluation of the held-out participant ----
        test_loader = self._make_loader(
            test_dataset, shuffle=False, seed=seed, batch_size=training.batch_size
        )
        evaluation = self._evaluate(model, test_loader, criterion)
        ledger = self._build_ledger_rows(
            test_dataset, evaluation, fold=fold, seed=seed, checkpoint_sha=checkpoint_sha
        )
        logger.info(
            "%s: held-out %s scored accuracy=%.3f balanced_accuracy=%.3f macro_f1=%.3f "
            "on %d window(s); fold took %s",
            label,
            fold.test_subject,
            evaluation["accuracy"],
            evaluation["balanced_accuracy"],
            evaluation["macro_f1"],
            len(test_dataset),
            progress.format_duration(time.perf_counter() - started),
            extra={
                "test_subject": fold.test_subject,
                "test_accuracy": evaluation["accuracy"],
                "test_balanced_accuracy": evaluation["balanced_accuracy"],
                "test_macro_f1": evaluation["macro_f1"],
                "n_test": len(test_dataset),
            },
        )

        outcome = FoldOutcome(
            outer_fold=fold.fold_index,
            test_subject=fold.test_subject,
            seed=seed,
            epoch_budget=epoch_budget,
            selected_hyperparameters=best_trial.hyperparameters,
            inner_trials=all_trials,
            training_history=[record.to_dict() for record in history],
            history_scope="refit_training_fit",
            normalizer=normalizer.to_dict(),
            class_weights=class_weights.tolist(),
            checkpoint_path=(
                checkpoint_path.relative_to(self.run_dir).as_posix() if checkpoint_path else None
            ),
            checkpoint_sha256=checkpoint_sha,
            parameter_counts=model.parameter_counts(),
            n_train=len(train_dataset),
            n_test=len(test_dataset),
            train_seconds=time.perf_counter() - started,
            augmentation_stats=(train_dataset.augmenter.stats if train_dataset.augmenter else {}),
        )
        return outcome, ledger

    def _build_ledger_rows(
        self,
        dataset: WindowDataset,
        evaluation: dict[str, Any],
        *,
        fold: OuterFold,
        seed: int,
        checkpoint_sha: str | None,
    ) -> pd.DataFrame:
        """Assemble the prediction ledger rows for one fold."""
        order = evaluation["indices"]
        probabilities = evaluation["probabilities"]
        y_true = evaluation["y_true"]
        y_pred = evaluation["y_pred"]

        windows = {w.sample_id: w for w in self.window_index.windows}
        rows: list[dict[str, Any]] = []
        for position, dataset_index in enumerate(order):
            sample_id = dataset.sample_ids[int(dataset_index)]
            window = windows[sample_id]
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "subject_id": window.subject_id,
                "recording_id": window.recording_id,
                "start_sample": window.start_sample,
                "end_sample": window.end_sample,
                "start_seconds": window.start_seconds,
                "end_seconds": window.end_seconds,
                "condition": window.condition,
                "task_family": window.task_family,
                "segment_source": window.segment_source,
                "true_label": int(y_true[position]),
                "predicted_label": int(y_pred[position]),
                "true_class": self.task.class_names[int(y_true[position])],
                "predicted_class": self.task.class_names[int(y_pred[position])],
                "outer_fold": fold.fold_index,
                "seed": seed,
                "task_id": self.task.task_id,
                "model_id": self.model_id,
                "modality": self.modality,
                "source_commit": self.source_commit,
                "config_hash": self.config.config_hash,
                "manifest_hash": self.manifest_hash,
                "window_index_hash": self.window_index.index_hash,
                "checkpoint_sha256": checkpoint_sha,
                "safe_for_inference": self.window_index.safe_for_inference,
            }
            for class_index, name in enumerate(self.task.class_names):
                row[f"prob_{name}"] = float(probabilities[position, class_index])
            rows.append(row)

        frame = pd.DataFrame(rows)
        missing = [column for column in LEDGER_COLUMNS if column not in frame.columns]
        if missing:
            raise AssertionError(f"prediction ledger is missing columns {missing}")
        return frame

    def run(
        self, *, seeds: Sequence[int] | None = None
    ) -> Iterator[tuple[FoldOutcome, pd.DataFrame]]:
        """Iterate every (outer fold, seed) pair, yielding results as they complete."""
        splitter = self._splitter()
        for seed in seeds if seeds is not None else self.config.training.seeds:
            for fold in splitter.outer_folds():
                yield self.run_fold(fold, seed)

    def _splitter(self) -> NestedLOSOSplitter:
        """Splitter with the calibration block withheld from every fold, if there is one."""
        return NestedLOSOSplitter(
            self.window_index,
            subjects=self.config.data.subjects,
            exclude_sample_ids=(
                self.calibration_block.all_sample_ids if self.calibration_block else frozenset()
            ),
        )

    def split_plan(self) -> dict[str, Any]:
        """The fold plan, written to ``folds.json`` before any training starts."""
        return self._splitter().to_dict()
