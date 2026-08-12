"""Run orchestration: config in, immutable run bundle out.

A run bundle is self-describing. Given only ``outputs/runs/<run_id>/`` a reviewer can read
the resolved configuration, the environment, the data manifest and its hash, the source
commit (and whether the tree was dirty), the fold plan, every hyperparameter trial, the
per-fold training history, the checkpoints, the full prediction ledger and the metrics
derived from it.

Execution is resumable at the granularity of a (condition, seed, outer fold) triple:
completed fold bundles are written before the next fold starts, and a resumed run skips
folds whose recorded configuration hash, manifest hash and window-index hash all match.
A mismatch is refused rather than silently mixed.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bruxism.config import ExperimentConfig
from bruxism.data.dataset import RecordingCache
from bruxism.data.labels import get_task
from bruxism.data.manifest import DatasetManifest, build_manifest
from bruxism.data.segments import WindowIndex, build_window_index
from bruxism.evaluation.aggregation import summarise_ledger
from bruxism.evaluation.metrics import PredictionLedger
from bruxism.models.ablations import AblationSpec
from bruxism.preprocessing.calibration import build_calibration_block
from bruxism.preprocessing.filters import chain_magnitude
from bruxism.preprocessing.wavelets import assert_bands_within_passband
from bruxism.training.engine import NestedLOSOTrainer, resolve_device
from bruxism.utils import progress
from bruxism.utils.io import (
    read_json,
    write_csv,
    write_json,
    write_parquet,
    write_yaml,
)
from bruxism.utils.logging import get_logger, setup_logging
from bruxism.utils.reproducibility import collect_environment, collect_source_state

__all__ = [
    "DataDefectError",
    "RunBundle",
    "assert_bands_are_inside_their_passband",
    "assert_modality_is_supported_by_data",
    "prepare_data",
    "run_experiment",
]

logger = get_logger(__name__)

RUN_BUNDLE_VERSION = "1.0"


@dataclass
class RunBundle:
    """Paths and identity of one run's output directory."""

    run_dir: Path
    run_id: str
    config_hash: str
    manifest_hash: str
    window_index_hash: str

    @property
    def predictions_path(self) -> Path:
        return self.run_dir / "predictions.parquet"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_bundle_version": RUN_BUNDLE_VERSION,
            "run_id": self.run_id,
            "run_dir": self.run_dir.as_posix(),
            "config_hash": self.config_hash,
            "manifest_hash": self.manifest_hash,
            "window_index_hash": self.window_index_hash,
        }


def prepare_data(
    config: ExperimentConfig, *, data_root: Path | str | None = None, probe_video: bool = False
) -> tuple[DatasetManifest, WindowIndex, RecordingCache]:
    """Build the manifest, window index and recording cache for an experiment.

    Parameters
    ----------
    data_root
        Overrides ``config.data.data_root``. Supplied by ``--data-root`` so no source file
        ever contains a machine-specific path.
    probe_video
        Video probing costs a few seconds and is unnecessary for training.
    """
    root = Path(data_root) if data_root is not None else Path(config.data.data_root)
    manifest = build_manifest(
        root,
        sampling_rate_hz=config.data.sampling_rate_hz,
        policy=config.exclusion_policy(),
        probe_video=probe_video,
    )
    window_index = build_window_index(manifest, config.data.segmentation())
    cache = RecordingCache(
        manifest,
        config.filters,
        cache_dir=config.data.cache_dir,
        emg_channels=config.data.emg_channels,
    )
    return manifest, window_index, cache


class DataDefectError(RuntimeError):
    """Raised when a run would consume a channel the manifest has flagged as defective."""


def assert_modality_is_supported_by_data(
    config: ExperimentConfig, manifest: DatasetManifest
) -> None:
    """Refuse a microphone-reading run on data whose microphone is flagged.

    The audit in ``audio.md`` found that 83 of the 100 recordings in this collection carry
    a microphone waveform belonging to another participant, that the channel is unaligned
    with the EMG, and that its analysis band is at the quantisation floor. A fusion or
    audio-only run on such data is not evaluating leave-one-subject-out on that channel,
    and the resulting modality contrast measures the defect rather than the microphone.

    Nothing here excludes a recording or changes a number. It stops a run that would
    produce an uninterpretable one, and it can be overridden -- in writing, in the
    configuration, recorded in the run bundle -- by
    :attr:`~bruxism.config.ExperimentConfig.mic_defect_acknowledged_by`. Reproducing the
    published ablation is exactly the case the override exists for.

    Raises
    ------
    DataDefectError
        If the run reads the microphone, the manifest flags it, and no acknowledgement is
        declared.
    """
    if not config.reads_microphone:
        return
    blocked = manifest.audio_blocking_flags()
    if not blocked:
        return

    reasons: dict[str, int] = {}
    for flags in blocked.values():
        for flag in flags:
            reasons[flag] = reasons.get(flag, 0) + 1
    summary = ", ".join(f"{flag} x{count}" for flag, count in sorted(reasons.items()))

    if config.mic_defect_acknowledged_by:
        logger.warning(
            "microphone-reading run proceeding on flagged data by declaration %r: "
            "%d of %d included recordings flagged (%s). Any modality contrast from this "
            "run describes the defect as much as the microphone -- see audio.md.",
            config.mic_defect_acknowledged_by,
            len(blocked),
            len(manifest.included),
            summary,
            extra={
                "acknowledged_by": config.mic_defect_acknowledged_by,
                "n_blocked": len(blocked),
                "reasons": reasons,
            },
        )
        return

    raise DataDefectError(
        f"experiment {config.name!r} has modality {config.modality!r} "
        f"(sweep: {list(config.sweep_modalities) or [config.modality]}) and therefore reads "
        f"the microphone, but {len(blocked)} of {len(manifest.included)} included recordings "
        f"carry a microphone-integrity flag: {summary}. See audio.md and rule "
        f"R4_mic_channel_is_not_analysable_audio. Run EMG-only, or -- if you are "
        f"deliberately reproducing a published result on this channel -- set "
        f"mic_defect_acknowledged_by: '<name>, <date>: <reason>' in the experiment "
        f"configuration so the concession is recorded in the run bundle."
    )


def assert_bands_are_inside_their_passband(config: ExperimentConfig) -> dict[str, dict[str, float]]:
    """Refuse a branch configured to read a band its own filter chain has removed.

    The mirror of ``cause.md``'s lesson on the wavelet side: there, a filter was verified
    against its own design instead of against the data; here, a band list was chosen
    without checking it against the chain in front of it. In this project both met -- the
    microphone chain high-passes at 20 Hz and the microphone branch reads ``A5``, nominally
    0-18.75 Hz, so a third of that branch is fed the high-pass roll-off residue and the
    batch normalisation behind it rescales that residue to unit variance.

    Returns
    -------
    dict
        ``branch -> {band: mean power gain}``, written into the run bundle so the number is
        recorded even when the check passes.
    """
    from bruxism.models.dual_branch import DualBranchConfig

    model = DualBranchConfig.from_dict(
        {"num_classes": config.task.num_classes, **dict(config.model_overrides)}
    )
    rate = float(config.data.sampling_rate_hz)
    gains: dict[str, dict[str, float]] = {}
    for branch, modality in (("emg", "emg"), ("mic", "mic")):
        if branch == "mic" and not config.reads_microphone:
            continue
        if branch == "emg" and config.modality == "audio_only" and not config.sweep_modalities:
            continue
        branch_config = model.emg if branch == "emg" else model.mic
        gains[branch] = assert_bands_within_passband(
            branch_config.wavelet,
            rate,
            chain_magnitude(config.filters, rate, modality=modality),  # type: ignore[arg-type]
            context=f"{config.name}: {branch} branch",
            acknowledged=config.stopband_bands_acknowledged_by,
        )
    return gains


def _default_run_id(config: ExperimentConfig) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"{config.name}_{stamp}_{config.config_hash[:8]}"


def _conditions(config: ExperimentConfig) -> list[AblationSpec]:
    """Every (task, model, modality) condition this run should execute."""
    task_ids = list(config.sweep_task_ids) or [config.task_id]
    model_ids = list(config.sweep_model_ids) or [config.model_id]
    modalities = list(config.sweep_modalities) or [config.modality]
    return [
        AblationSpec(task_id=task_id, model_id=model_id, modality=modality)  # type: ignore[arg-type]
        for task_id in task_ids
        for model_id in model_ids
        for modality in modalities
    ]


def run_experiment(
    config: ExperimentConfig,
    *,
    data_root: Path | str | None = None,
    resume: bool = True,
    dry_run: bool = False,
    max_folds: int | None = None,
    figures: bool = True,
) -> RunBundle:
    """Execute an experiment and write its run bundle.

    Parameters
    ----------
    resume
        Skip folds already present in the bundle whose config, manifest and window-index
        hashes match. A mismatch raises instead of mixing incompatible artifacts.
    dry_run
        Build and validate everything, write the plan, and stop before training. Used by
        ``--validate-only`` to check a config against real data cheaply.
    max_folds
        Cap the number of (condition, seed, fold) triples executed. Used by the smoke run.
    figures
        Write the run's figure set into ``<run_dir>/figures`` once the last fold is done.
        On by default: a finished run should explain itself. Figure generation is
        deliberately not part of the configuration hash -- it produces no number a result
        depends on -- so turning it off cannot change what a run computes.

    Returns
    -------
    RunBundle
        Paths and identity of the produced bundle.
    """
    run_started = time.perf_counter()
    manifest, window_index, cache = prepare_data(config, data_root=data_root)
    # Both guards run before the run directory is created: a refused run must leave no
    # half-written bundle behind for a later resume to pick up.
    assert_modality_is_supported_by_data(config, manifest)
    band_gains = assert_bands_are_inside_their_passband(config)
    run_id = config.output.run_id or _default_run_id(config)
    run_dir = Path(config.output.runs_root) / run_id
    if run_dir.exists() and not (resume or config.output.overwrite):
        raise FileExistsError(
            f"run directory {run_dir} already exists; pass --resume to continue it or "
            f"set output.overwrite"
        )
    for sub in ("selection", "logs", "checkpoints", "figures", "folds"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=run_dir / "logs")

    source_state = collect_source_state()
    bundle = RunBundle(
        run_dir=run_dir,
        run_id=run_id,
        config_hash=config.config_hash,
        manifest_hash=manifest.manifest_hash,
        window_index_hash=window_index.index_hash,
    )

    write_yaml(run_dir / "resolved_config.yaml", config.to_dict())
    write_json(run_dir / "environment.json", collect_environment())
    write_json(run_dir / "source_state.json", source_state.to_dict())
    write_json(
        run_dir / "data_manifest.json",
        {
            "manifest_hash": manifest.manifest_hash,
            "quality_policy_version": manifest.policy.policy_version,
            "n_records": len(manifest.records),
            "n_included": len(manifest.included),
            "subject_ids": manifest.subject_ids,
            # Recorded whether or not this run reads the microphone, so a bundle can always
            # be asked what state the audio channel was in when it was produced.
            "channel_duplication": manifest.duplication,
            "audio_blocking_flags": manifest.audio_blocking_flags(),
            "mic_defect_acknowledged_by": config.mic_defect_acknowledged_by,
            "reads_microphone": config.reads_microphone,
            "wavelet_band_passband_gains": band_gains,
            "stopband_bands_acknowledged_by": config.stopband_bands_acknowledged_by,
            "segmentation": window_index.config.to_dict(),
            "window_index_hash": window_index.index_hash,
            "n_windows": len(window_index.windows),
            "window_counts_by_family": window_index.counts_by("task_family").to_dict("records"),
            "window_counts_by_subject_family": window_index.counts_by(
                "subject_id", "task_family"
            ).to_dict("records"),
        },
    )
    (run_dir / "data_manifest.sha256").write_text(manifest.manifest_hash + "\n", encoding="utf-8")
    write_csv(run_dir / "data_manifest.csv", manifest.frame)
    write_json(run_dir / "run_bundle.json", bundle.to_dict())

    if source_state.is_dirty:
        logger.warning(
            "run launched from a DIRTY working tree (%d modified files, diff sha %s); "
            "the exact source is not recoverable from the commit alone",
            len(source_state.dirty_files),
            source_state.diff_sha256,
        )

    # Per-participant normalisation needs a calibration block, and the block must be
    # withheld from every split -- so it is built before the splits, not inside training.
    calibration_block = None
    if config.normalization.scope == "per_participant":
        calibration_block = build_calibration_block(
            window_index,
            source=config.normalization.calibration,
            subjects=config.data.subjects,
        )
        write_json(run_dir / "calibration_block.json", calibration_block.to_dict())
        logger.warning(
            "TRANSDUCTIVE PROTOCOL: %d calibration window(s) withheld from every split and "
            "used to set each participant's own normalisation scale, including the "
            "held-out participant's. Report these results as requiring a fitting session.",
            len(calibration_block.all_sample_ids),
        )

    conditions = _conditions(config)
    plan: dict[str, Any] = {
        "run_id": run_id,
        "conditions": [spec.to_dict() for spec in conditions],
        "seeds": list(config.training.seeds),
        "safe_for_inference": window_index.safe_for_inference,
        "segmentation_policy": str(window_index.config.policy),
        "normalization": {
            **config.normalization.to_dict(),
            # Derived, so it lives here rather than in the round-tripped configuration.
            "is_transductive": config.normalization.is_transductive,
        },
        "calibration_block": calibration_block.to_dict() if calibration_block else None,
    }
    first_task = get_task(conditions[0].task_id)
    planner = NestedLOSOTrainer(
        config,
        window_index,
        cache,
        first_task,
        run_dir=run_dir,
        source_commit=source_state.commit,
        manifest_hash=manifest.manifest_hash,
        calibration_block=calibration_block,
    )
    plan["folds"] = planner.split_plan()
    write_json(run_dir / "folds.json", plan)

    if dry_run:
        logger.info("dry run complete; plan written to %s", run_dir / "folds.json")
        return bundle

    from bruxism.data.splits import NestedLOSOSplitter

    splitter = NestedLOSOSplitter(
        window_index,
        subjects=config.data.subjects,
        exclude_sample_ids=(calibration_block.all_sample_ids if calibration_block else frozenset()),
    )
    n_planned = len(conditions) * len(config.training.seeds) * splitter.n_outer_folds
    if max_folds is not None:
        n_planned = min(n_planned, max_folds)
    # `_trial_grid` takes the Cartesian product of the search space; an empty space is one
    # trial. Recomputed here only to state the size of the job before it starts.
    n_trials = math.prod(len(values) for values in config.training.search_space.values())
    logger.info(
        "plan: %d condition(s) x %d seed(s) x %d outer fold(s) = %d fold run(s); "
        "%d model fit(s) per fold (%d trial(s) x %d inner fold(s), then 1 refit); device=%s",
        len(conditions),
        len(config.training.seeds),
        splitter.n_outer_folds,
        n_planned,
        n_trials * splitter.n_inner_folds() + 1,
        n_trials,
        splitter.n_inner_folds(),
        resolve_device(config.training.device).type,
        extra={"n_planned_folds": n_planned, "n_trials": n_trials},
    )

    executed = 0
    trained_seconds: list[float] = []
    ledgers: list[pd.DataFrame] = []
    outcomes: list[dict[str, Any]] = []

    with progress.task("folds", total=n_planned, unit="fold", leave=True) as fold_progress:
        for spec in conditions:
            task = get_task(spec.task_id)
            trainer = NestedLOSOTrainer(
                config,
                window_index,
                cache,
                task,
                run_dir=run_dir,
                source_commit=source_state.commit,
                manifest_hash=manifest.manifest_hash,
                model_id=spec.model_id,
                modality=spec.modality,
                calibration_block=calibration_block,
            )
            for seed in config.training.seeds:
                for fold in splitter.outer_folds():
                    if max_folds is not None and executed >= max_folds:
                        logger.warning(
                            "stopping after %d fold(s) because max_folds=%d was set; this run "
                            "is INCOMPLETE and its metrics cover only the folds executed",
                            executed,
                            max_folds,
                        )
                        break
                    slug = (
                        f"{spec.task_id}__{spec.model_id}__{spec.modality}"
                        f"__seed{seed}__fold{fold.fold_index}"
                    )
                    label = f"fold {executed + 1}/{n_planned}"
                    fold_progress.set_description(f"{label} {spec.task_id} {fold.test_subject}")
                    fold_json = run_dir / "folds" / f"{slug}.json"
                    fold_parquet = run_dir / "folds" / f"{slug}.parquet"
                    if resume and fold_json.is_file() and fold_parquet.is_file():
                        stored = read_json(fold_json)
                        _assert_compatible(stored, bundle, slug)
                        logger.info("resume: reusing completed fold %s", slug)
                        ledgers.append(pd.read_parquet(fold_parquet))
                        outcomes.append(stored)
                        executed += 1
                        fold_progress.update(1)
                        continue

                    fold_started = time.perf_counter()
                    outcome, ledger = trainer.run_fold(fold, seed, progress_label=label)
                    trained_seconds.append(time.perf_counter() - fold_started)
                    payload = outcome.to_dict()
                    payload.update(
                        {
                            "condition": spec.to_dict(),
                            "config_hash": bundle.config_hash,
                            "manifest_hash": bundle.manifest_hash,
                            "window_index_hash": bundle.window_index_hash,
                            "run_bundle_version": RUN_BUNDLE_VERSION,
                        }
                    )
                    # Persist BEFORE moving on, so a crash never loses a completed fold.
                    write_parquet(fold_parquet, ledger)
                    write_json(fold_json, payload)
                    ledgers.append(ledger)
                    outcomes.append(payload)
                    executed += 1
                    fold_progress.update(1)
                    logger.info(
                        "%s complete in %s; %d of %d fold(s) done, ~%s remaining",
                        label,
                        progress.format_duration(trained_seconds[-1]),
                        executed,
                        n_planned,
                        _remaining_estimate(trained_seconds, n_planned - executed),
                        extra={"folds_done": executed, "folds_planned": n_planned},
                    )
                else:
                    continue
                break

    if not ledgers:
        raise RuntimeError("no folds were executed; nothing to report")

    predictions = pd.concat(ledgers, ignore_index=True).sort_values(
        ["task_id", "model_id", "modality", "seed", "outer_fold", "sample_id"],
        ignore_index=True,
    )
    write_parquet(bundle.predictions_path, predictions)
    write_json(run_dir / "selection" / "fold_outcomes.json", outcomes)

    metrics = summarise_ledger(predictions)
    write_json(bundle.metrics_path, metrics)
    write_csv(run_dir / "metrics.csv", _metrics_to_frame(metrics))

    if figures:
        _write_figures(
            bundle,
            config=config,
            manifest=manifest,
            window_index=window_index,
            cache=cache,
            predictions=predictions,
            metrics=metrics,
            outcomes=outcomes,
        )

    logger.info(
        "run %s complete in %s: %d folds, %d predictions -> %s",
        run_id,
        progress.format_duration(time.perf_counter() - run_started),
        executed,
        len(predictions),
        run_dir,
    )
    return bundle


def _write_figures(
    bundle: RunBundle,
    *,
    config: ExperimentConfig,
    manifest: DatasetManifest,
    window_index: WindowIndex,
    cache: RecordingCache,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    outcomes: Sequence[dict[str, Any]],
) -> None:
    """Draw the run's figures, reusing the data already loaded for training.

    Never raises: the folds, ledger and metrics are already written and durable by this
    point, so a plotting failure must not turn a completed run into a failed command. The
    reason for any missing figure is recorded in ``figures/figure_index.json``.
    """
    from bruxism.visualization.run_figures import generate_run_figures

    started = time.perf_counter()
    try:
        summary = generate_run_figures(
            bundle.run_dir,
            config=config,
            manifest=manifest,
            window_index=window_index,
            cache=cache,
            predictions=predictions,
            metrics=metrics,
            fold_outcomes=list(outcomes),
        )
    except Exception as exc:  # noqa: BLE001 - a finished run must not fail on a figure
        logger.warning(
            "figure generation failed (%s: %s); the run bundle itself is complete and "
            "`bruxism-figures --run-dir %s` can retry it",
            type(exc).__name__,
            exc,
            bundle.run_dir,
            exc_info=True,
        )
        return
    logger.info(
        "figures written in %s: %d figure(s) in %s (%d skipped, %d failed)",
        progress.format_duration(time.perf_counter() - started),
        summary["written"],
        summary["figures_dir"],
        summary["skipped"],
        summary["failed"],
    )


def _remaining_estimate(trained_seconds: Sequence[float], folds_left: int) -> str:
    """Wall-clock estimate for the folds still to train, from the ones already trained.

    Resumed folds cost nothing and are deliberately excluded from the average, so a
    resumed run does not report an optimistic estimate for the work it has left.
    """
    if folds_left <= 0:
        return "0s"
    if not trained_seconds:
        return "unknown"
    return progress.format_duration(sum(trained_seconds) / len(trained_seconds) * folds_left)


def _assert_compatible(stored: dict[str, Any], bundle: RunBundle, slug: str) -> None:
    """Refuse to resume across an incompatible configuration or dataset."""
    for key, expected in (
        ("config_hash", bundle.config_hash),
        ("manifest_hash", bundle.manifest_hash),
        ("window_index_hash", bundle.window_index_hash),
    ):
        actual = stored.get(key)
        if actual != expected:
            raise ValueError(
                f"cannot resume fold {slug}: stored {key}={actual!r} but this run has "
                f"{expected!r}. Delete the run directory or use a new run_id rather than "
                f"mixing incompatible artifacts."
            )


def _metrics_to_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    """Flatten the headline numbers of each condition into a single CSV row per condition."""
    rows: list[dict[str, Any]] = []
    for condition_id, payload in metrics.get("conditions", {}).items():
        subject_level = payload.get("subject_level", {})
        pooled = payload.get("pooled_windows", {})
        row: dict[str, Any] = {
            "condition_id": condition_id,
            "task_id": payload.get("task_id"),
            "model_id": payload.get("model_id"),
            "modality": payload.get("modality"),
            "seed": payload.get("seed"),
            "n_subjects": subject_level.get("n_subjects"),
            "n_windows": pooled.get("n_samples"),
            "safe_for_inference": payload.get("safe_for_inference"),
        }
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "macro_roc_auc"):
            entry = subject_level.get(metric)
            if isinstance(entry, dict):
                row[f"subject_{metric}_mean"] = entry.get("mean")
                row[f"subject_{metric}_std"] = entry.get("std")
                row[f"subject_{metric}_min"] = entry.get("min")
                row[f"subject_{metric}_max"] = entry.get("max")
        row["pooled_accuracy_descriptive"] = pooled.get("accuracy")
        row["pooled_macro_f1_descriptive"] = pooled.get("macro_f1")
        row["pooled_macro_roc_auc_descriptive"] = pooled.get("macro_roc_auc")
        rows.append(row)
    return pd.DataFrame(rows)


def load_predictions(run_dir: Path | str) -> tuple[pd.DataFrame, list[str]]:
    """Load a run's prediction ledger and infer its class-name order."""
    frame = pd.read_parquet(Path(run_dir) / "predictions.parquet")
    class_names = [c.removeprefix("prob_") for c in frame.columns if c.startswith("prob_")]
    return frame, class_names


def ledger_for(frame: pd.DataFrame, task_id: str) -> PredictionLedger:
    """Build a :class:`PredictionLedger` for one task from a mixed-task frame."""
    task = get_task(task_id)
    subset = frame[frame["task_id"] == task_id].reset_index(drop=True)
    return PredictionLedger(frame=subset, class_names=task.class_names)
