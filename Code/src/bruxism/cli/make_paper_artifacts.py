"""``bruxism-report`` -- regenerate the entire manuscript bundle from saved artifacts.

One command produces every figure, table, LaTeX macro and the narrative results document::

    bruxism-report --runs-root outputs/runs --output-root outputs/paper_bundle

Everything is derived from ``predictions.parquet``, ``metrics.json``,
``selection/fold_outcomes.json`` and the optional benchmark JSON. The only step that needs
the raw data is the exploratory t-SNE, which must recompute held-out embeddings from a
saved checkpoint; it is skipped with a recorded reason when ``--data-root`` is absent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from bruxism.cli._common import (
    add_common_arguments,
    add_data_root_argument,
    make_parser,
    parse_and_run,
    resolve_data_root,
)
from bruxism.cli.summarize_runs import collect_predictions
from bruxism.data.labels import get_task
from bruxism.evaluation.aggregation import condition_table, summarise_ledger
from bruxism.evaluation.metrics import PredictionLedger
from bruxism.models import BruxismModel
from bruxism.reporting import render_paper_results
from bruxism.utils.io import read_json, write_csv, write_json
from bruxism.utils.logging import get_logger
from bruxism.visualization.paper_figures import (
    plot_confusion_matrices,
    plot_flow_diagram,
    plot_modality_comparison,
    plot_per_participant,
    plot_pr_curves,
    plot_roc_curves,
    plot_training_curves,
    plot_tsne,
    write_latex_macros,
    write_latex_tables,
)

logger = get_logger(__name__)

#: Placeholders in ``Main_2.tex`` that this bundle is responsible for supplying, mapped to
#: the artifact that supplies them. Anything not listed here needs a human decision.
MANUSCRIPT_ASSET_MAP: dict[str, str] = {
    "Figures/confmatrx_5class.png": "figures/five_class_confusion_matrix.png",
    "Figures/tsne_5class.png": "figures/five_class_tsne.png",
    "Figures/pipeline_5class.png": "BLOCKED: architecture diagram is drawn by hand",
    "training accuracy curves": "figures/training_curves.png",
    "training loss curves": "figures/training_curves.png",
    "ROC / AUC figure": "figures/five_class_roc_curves.png",
    "PR / average-precision figure": "figures/five_class_pr_curves.png",
    "per-participant results": "figures/five_class_per_participant.png",
    "modality comparison": "figures/modality_comparison.png",
    "sample counts / flow": "figures/sample_flow.png",
    "parameter count and latency": "tables/benchmark.csv",
}


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("bruxism-report", __doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--run-id", action="append", default=[], help="Repeatable.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_bundle"))
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("outputs/benchmarks/benchmark.json"),
        help="Benchmark JSON from bruxism-benchmark. Skipped if absent.",
    )
    parser.add_argument(
        "--primary-task", default="five_class", help="Task that drives the headline figures."
    )
    add_data_root_argument(parser)
    parser.add_argument(
        "--tsne-seed", type=int, default=0, help="Fixed seed for the exploratory t-SNE."
    )
    parser.add_argument(
        "--tsne-max-samples",
        type=int,
        default=3000,
        help="Deterministic evenly spaced subsample cap for the t-SNE.",
    )
    parser.add_argument("--no-tsne", action="store_true", help="Skip the t-SNE figure.")
    return parser


def _fold_outcomes(runs_root: Path, run_ids: list[str]) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for run_id in run_ids:
        path = runs_root / run_id / "selection" / "fold_outcomes.json"
        if path.is_file():
            outcomes.extend(read_json(path))
    return outcomes


def _provenance(runs_root: Path, run_ids: list[str]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for run_id in run_ids:
        source = runs_root / run_id / "source_state.json"
        bundle = runs_root / run_id / "run_bundle.json"
        if source.is_file():
            state = read_json(source)
            entries[f"{run_id}:commit"] = state.get("commit")
            entries[f"{run_id}:dirty"] = state.get("is_dirty")
        if bundle.is_file():
            payload = read_json(bundle)
            entries[f"{run_id}:manifest_hash"] = payload.get("manifest_hash")
            entries[f"{run_id}:config_hash"] = payload.get("config_hash")
    return entries


def _compute_embeddings(
    runs_root: Path,
    run_ids: list[str],
    task_id: str,
    data_root: Path,
    *,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, list[str], str] | None:
    """Recompute held-out embeddings from saved checkpoints for the t-SNE figure.

    Uses only outer-test windows, so the projection shows genuinely held-out data.
    """
    import torch

    from bruxism.config import load_experiment_config
    from bruxism.data.dataset import WindowDataset
    from bruxism.models.baselines import build_neural_model
    from bruxism.preprocessing.normalization import Normalizer
    from bruxism.runner import prepare_data

    for run_id in run_ids:
        run_dir = runs_root / run_id
        config_path = run_dir / "resolved_config.yaml"
        checkpoints = sorted((run_dir / "checkpoints").glob(f"{task_id}_*.pt"))
        if not config_path.is_file() or not checkpoints:
            continue
        config = load_experiment_config(config_path)
        _, window_index, cache = prepare_data(config, data_root=data_root)
        task = get_task(task_id)

        predictions = pd.read_parquet(run_dir / "predictions.parquet")
        predictions = predictions[predictions["task_id"] == task_id]

        embeddings: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        used: list[str] = []
        for checkpoint_path in checkpoints:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            architecture = payload["architecture"]
            model = build_neural_model(
                architecture["config"].get("model_id", config.model_id)
                if "model_id" in architecture["config"]
                else config.model_id,
                num_classes=task.num_classes,
                modality=architecture["config"].get("modality", config.modality),
                emg_channels=cache.n_emg_channels,
                window_samples=config.data.window_samples(),
            )
            model.load_state_dict(payload["state_dict"])
            model.eval()

            normalizer = Normalizer.from_dict(payload["normalizer"])
            fold_subjects = set(normalizer.fitted_on)
            held_out = sorted(set(predictions["subject_id"]) - fold_subjects)
            fold_rows = predictions[predictions["subject_id"].isin(held_out)]
            sample_ids = list(fold_rows["sample_id"].unique())
            if not sample_ids:
                continue
            dataset = WindowDataset(
                window_index,
                sample_ids,
                task,
                cache,
                stage="test",
                normalizer=normalizer,
                modality=architecture["config"].get("modality", config.modality),
            )
            with torch.no_grad():
                for start in range(0, len(dataset), 128):
                    batch = [dataset[i] for i in range(start, min(start + 128, len(dataset)))]
                    emg = torch.stack([item[0] for item in batch])
                    mic = torch.stack([item[1] for item in batch])
                    embeddings.append(cast(BruxismModel, model).embed(emg, mic).numpy())
                    labels.append(np.array([int(item[2]) for item in batch]))
            used.append(checkpoint_path.name)

        if embeddings:
            stacked = np.concatenate(embeddings, axis=0)
            label_array = np.concatenate(labels, axis=0)
            if stacked.shape[0] > max_samples:
                step = np.linspace(0, stacked.shape[0] - 1, max_samples).astype(int)
                stacked, label_array = stacked[step], label_array[step]
            return stacked, label_array, list(task.class_names), ", ".join(used)
    return None


def main_impl(args: argparse.Namespace) -> int:
    predictions, run_ids = collect_predictions(args.runs_root, args.run_id)
    output_root = Path(args.output_root)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    output_root.mkdir(parents=True, exist_ok=True)

    metrics = summarise_ledger(predictions)
    write_json(output_root / "metrics.json", metrics)
    table = condition_table(metrics)
    write_csv(output_root / "condition_table.csv", table)

    artifact_index: dict[str, str] = {
        "prediction ledger (concatenated)": "predictions.parquet",
        "metrics (recomputed from the ledger)": "metrics.json",
        "condition comparison table": "condition_table.csv",
    }
    predictions.to_parquet(output_root / "predictions.parquet", index=False)

    tasks_present = sorted(predictions["task_id"].unique())
    primary = (
        args.primary_task
        if args.primary_task in tasks_present
        else (tasks_present[0] if tasks_present else None)
    )
    blockers: list[dict[str, str]] = []
    macros: dict[str, Any] = {}

    # ---- per-task figures ----
    for task_id in tasks_present:
        task = get_task(task_id)
        subset = predictions[predictions["task_id"] == task_id]
        # One representative condition per task for the confusion/curve figures: the
        # fusion condition of the dual-branch model at the lowest seed, chosen by a fixed
        # rule rather than by which looked best.
        preferred = subset[
            (subset["modality"] == "fusion") & (subset["model_id"] == "dual_branch_wavelet_cnn")
        ]
        chosen = preferred if not preferred.empty else subset
        chosen = chosen[chosen["seed"] == chosen["seed"].min()]
        ledger = PredictionLedger(frame=chosen.reset_index(drop=True), class_names=task.class_names)
        stem = task_id
        plot_confusion_matrices(ledger, figures_dir, stem=f"{stem}_confusion_matrix")
        artifact_index[f"{task_id}: confusion matrices"] = f"figures/{stem}_confusion_matrix.png"

        condition_key = next(
            (
                key
                for key, entry in metrics["conditions"].items()
                if entry["task_id"] == task_id
                and entry["modality"] == chosen["modality"].iloc[0]
                and entry["model_id"] == chosen["model_id"].iloc[0]
                and entry["seed"] == int(chosen["seed"].iloc[0])
            ),
            None,
        )
        if condition_key:
            entry = metrics["conditions"][condition_key]
            curves = entry.get("curves", {})
            if curves:
                plot_roc_curves(
                    curves,
                    figures_dir,
                    stem=f"{stem}_roc_curves",
                    title=f"{task_id}: one-vs-rest ROC",
                )
                plot_pr_curves(
                    curves,
                    figures_dir,
                    stem=f"{stem}_pr_curves",
                    title=f"{task_id}: one-vs-rest precision-recall",
                )
                artifact_index[f"{task_id}: ROC curves"] = f"figures/{stem}_roc_curves.png"
                artifact_index[f"{task_id}: PR curves"] = f"figures/{stem}_pr_curves.png"
            plot_per_participant(entry, figures_dir, stem=f"{stem}_per_participant")
            artifact_index[f"{task_id}: per-participant"] = f"figures/{stem}_per_participant.png"
            if task_id == primary:
                subject = entry["subject_level"]
                for metric in ("accuracy", "balanced_accuracy", "macro_f1", "macro_roc_auc"):
                    block = subject.get(metric)
                    if isinstance(block, dict):
                        macros[f"{metric}_mean"] = block.get("mean")
                        macros[f"{metric}_std"] = block.get("std")
                macros["n_participants"] = subject.get("n_subjects")
                macros["n_held_out_windows"] = entry["pooled_windows"].get("n_samples")

        # per-class table
        pooled = metrics["conditions"].get(condition_key, {}).get("pooled_windows", {})
        if pooled.get("per_class"):
            per_class = pd.DataFrame(
                [{"class": name, **values} for name, values in pooled["per_class"].items()]
            )
            auc = pooled.get("roc_auc_per_class", {})
            ap = pooled.get("average_precision_per_class", {})
            per_class["roc_auc"] = per_class["class"].map(auc)
            per_class["average_precision"] = per_class["class"].map(ap)
            write_latex_tables(
                {
                    f"{task_id}_per_class": (
                        per_class,
                        f"Per-class held-out performance for the {task_id} task "
                        "(pooled windows; descriptive).",
                    )
                },
                tables_dir,
            )
            artifact_index[f"{task_id}: per-class table"] = f"tables/{task_id}_per_class.csv"

    # ---- cross-task figures ----
    if not table.empty and table["modality"].nunique() > 1:
        plot_modality_comparison(table, figures_dir)
        artifact_index["modality comparison"] = "figures/modality_comparison.png"
    else:
        blockers.append(
            {
                "item": "modality comparison figure",
                "reason": "only one modality present in the ledger",
                "action": "run bruxism-ablations with configs/experiments/modality_and_no_chewing.yaml",
            }
        )

    outcomes = _fold_outcomes(args.runs_root, run_ids)
    if outcomes:
        plot_training_curves(outcomes, figures_dir)
        artifact_index["training curves"] = "figures/training_curves.png"
        selection = pd.DataFrame(
            [
                {
                    "outer_fold": o.get("outer_fold"),
                    "held_out_subject": o.get("test_subject"),
                    "seed": o.get("seed"),
                    "epoch_budget": o.get("epoch_budget"),
                    "n_train_windows": o.get("n_train"),
                    "n_test_windows": o.get("n_test"),
                    "selected_hyperparameters": str(o.get("selected_hyperparameters")),
                    "train_seconds": o.get("train_seconds"),
                }
                for o in outcomes
            ]
        )
        write_latex_tables(
            {"selection_summary": (selection, "Nested LOSO selection summary per outer fold.")},
            tables_dir,
        )
        artifact_index["selection summary"] = "tables/selection_summary.csv"

    counts = (
        predictions[predictions["task_id"] == primary]
        .drop_duplicates("sample_id")
        .groupby(["subject_id", "task_family"], observed=True)
        .size()
        .reset_index(name="n_windows")
        if primary and "task_family" in predictions.columns
        else pd.DataFrame()
    )
    if not counts.empty:
        plot_flow_diagram(counts, figures_dir)
        write_csv(tables_dir / "sample_counts.csv", counts)
        artifact_index["sample flow"] = "figures/sample_flow.png"

    # ---- benchmark ----
    benchmark: dict[str, Any] | None = None
    if args.benchmark and Path(args.benchmark).is_file():
        payload = read_json(args.benchmark)
        # Select the model that produced the headline figures, not whichever entry happens
        # to come first in the JSON.
        primary_model = (
            str(predictions[predictions["task_id"] == primary]["model_id"].iloc[0])
            if primary and not predictions[predictions["task_id"] == primary].empty
            else None
        )
        if isinstance(payload, dict) and payload:
            benchmark = payload.get(primary_model) or next(iter(payload.values()))
            if primary_model and primary_model not in payload:
                logger.warning(
                    "benchmark JSON has no entry for the headline model %r; using %r "
                    "instead. Re-run bruxism-benchmark --models %s for a matching number.",
                    primary_model,
                    next(iter(payload)),
                    primary_model,
                )
        csv_path = Path(args.benchmark).with_suffix(".csv")
        if csv_path.is_file():
            write_csv(tables_dir / "benchmark.csv", pd.read_csv(csv_path))
            artifact_index["benchmark table"] = "tables/benchmark.csv"
        if benchmark:
            macros["benchmarked_model"] = benchmark.get("config", {}).get("model_id", primary_model)
            macros["trainable_parameters"] = benchmark["parameter_counts"].get("trainable")
            budget = benchmark["latency_budget"]
            macros["processing_latency_ms"] = budget["processing_latency_seconds"] * 1e3
            macros["input_context_latency_ms"] = budget["input_context_latency_seconds"] * 1e3
            macros["decision_interval_ms"] = budget["decision_update_interval_seconds"] * 1e3
    else:
        blockers.append(
            {
                "item": "parameter count and latency table",
                "reason": f"no benchmark JSON at {args.benchmark}",
                "action": "run bruxism-benchmark --output outputs/benchmarks",
            }
        )

    # ---- exploratory t-SNE ----
    if args.no_tsne:
        blockers.append(
            {"item": "t-SNE figure", "reason": "--no-tsne", "action": "rerun without --no-tsne"}
        )
    elif primary is None:
        blockers.append(
            {
                "item": "t-SNE figure",
                "reason": "no task in the ledger",
                "action": "run an experiment",
            }
        )
    else:
        data_root = resolve_data_root(args.data_root, required=False)
        if data_root is None:
            blockers.append(
                {
                    "item": "t-SNE figure",
                    "reason": "needs --data-root to recompute held-out embeddings from a checkpoint",
                    "action": "rerun bruxism-report with --data-root /path/to/Data",
                }
            )
        else:
            result = _compute_embeddings(
                args.runs_root, run_ids, primary, data_root, max_samples=args.tsne_max_samples
            )
            if result is None:
                blockers.append(
                    {
                        "item": "t-SNE figure",
                        "reason": "no usable checkpoint found in the selected runs",
                        "action": "rerun training with output.save_checkpoints: true",
                    }
                )
            else:
                embeddings, labels, class_names, checkpoint_id = result
                plot_tsne(
                    embeddings,
                    labels,
                    class_names,
                    figures_dir,
                    seed=args.tsne_seed,
                    checkpoint_id=checkpoint_id,
                    stem=f"{primary}_tsne",
                )
                artifact_index[f"{primary}: exploratory t-SNE"] = f"figures/{primary}_tsne.png"

    # ---- LaTeX ----
    write_latex_tables(
        {"condition_comparison": (table, "Comparison across tasks, models and modalities.")},
        tables_dir,
    )
    artifact_index["condition comparison (LaTeX)"] = "tables/condition_comparison.tex"
    macro_path = write_latex_macros(macros, tables_dir)
    artifact_index["LaTeX macros"] = f"tables/{macro_path.name}"

    write_json(output_root / "manuscript_asset_map.json", MANUSCRIPT_ASSET_MAP)
    artifact_index["manuscript asset map"] = "manuscript_asset_map.json"

    provenance = _provenance(args.runs_root, run_ids)
    manifest_hash = next(
        (v for k, v in provenance.items() if k.endswith(":manifest_hash")), "unknown"
    )
    commit = next((v for k, v in provenance.items() if k.endswith(":commit")), None)
    report = render_paper_results(
        metrics,
        run_ids=run_ids,
        artifact_index=artifact_index,
        manifest_hash=str(manifest_hash),
        source_commit=commit,
        blockers=blockers,
        benchmark=benchmark,
        sample_counts=counts if not counts.empty else None,
        provenance=provenance,
    )
    (output_root / "paper_results.md").write_text(report, encoding="utf-8")

    print(f"paper bundle : {output_root}")
    print(f"figures      : {len(list(figures_dir.glob('*.png')))} PNG")
    print(
        f"tables       : {len(list(tables_dir.glob('*.csv')))} CSV / "
        f"{len(list(tables_dir.glob('*.tex')))} TeX"
    )
    if blockers:
        print(f"\n{len(blockers)} blocked artifact(s):")
        for blocker in blockers:
            print(f"  - {blocker['item']}: {blocker['reason']}")
            print(f"    -> {blocker['action']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return parse_and_run(build_parser, main_impl, argv)


if __name__ == "__main__":
    raise SystemExit(main())
