"""Regenerate the run-dependent manuscript figures for ``Paper/K_Farhadi_Paper_Bruxism``.

Seven of the manuscript's figures depend on this run and must be rebuilt whenever the
reported run changes (the modality-ablation figure depends on a different run and is drawn
by ``make_ablation_figure.py``)::

    Figures/confmatrx_5class            held-out confusion matrices
    Figures/five_class_roc_curves       one-vs-rest ROC
    Figures/five_class_pr_curves        one-vs-rest precision-recall
    Figures/five_class_per_participant  macro-F1 per held-out participant
    Figures/sample_flow                 analysable windows per participant and class
    Figures/training_curves_5class      refit loss and training-fit macro-F1
    Figures/tsne_5class                 t-SNE of held-out fusion embeddings

The remaining manuscript figures (sensor photo, filter defect and correction, preprocessing
stages, architecture diagram, signal comparison) depend on the data and the preprocessing
chain rather than on the training run, and are left alone.

Everything except the t-SNE is derived from the run bundle's saved artifacts --
``predictions.parquet`` and ``selection/fold_outcomes.json`` -- so a figure cannot disagree
with the ledger it depicts. The t-SNE is the one exception: it must recompute held-out
embeddings from the saved fold checkpoints, so it needs ``--data-root``.

Usage, from ``Code/``::

    python scripts/evaluate/make_manuscript_figures.py \
        --run-dir outputs/runs/five_class_nested_loso_20260807T211827_2b6fb5ac \
        --data-root ../Data
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bruxism.evaluation.metrics import (  # noqa: E402
    PredictionLedger,
    confusion_matrices,
    curve_points,
    subject_level_summary,
)
from bruxism.visualization.paper_figures import FigureStyle, save_figure  # noqa: E402

#: Ledger class identifiers in label order, with the wording the manuscript uses. The
#: figures must not show code identifiers such as ``instructed_grinding``.
CLASS_IDS: tuple[str, ...] = ("rest", "movement", "clench", "instructed_grinding", "chewing")
CLASS_LABELS: dict[str, str] = {
    "rest": "Rest",
    "movement": "Movement",
    "clench": "Clenching",
    "instructed_grinding": "Grinding",
    "chewing": "Chewing",
}
#: One colour per class, fixed across every figure so a reader can carry the mapping over.
CLASS_COLORS: dict[str, str] = {
    name: FigureStyle.color(index) for index, name in enumerate(CLASS_IDS)
}
#: Seed 0 solid, seed 1 dashed, seed 2 dotted -- the training-curve figure needs to show
#: fifteen curves without fifteen colours.
SEED_STYLES: tuple[str, ...] = ("-", "--", ":")

DEFAULT_OUTPUT = Path("../Paper/K_Farhadi_Paper_Bruxism/Figures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed run bundle.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Recording root. Required for the t-SNE, which recomputes held-out embeddings.",
    )
    parser.add_argument("--task-id", default="five_class")
    parser.add_argument(
        "--figure-seed",
        type=int,
        default=None,
        help=(
            "Seed for the confusion / ROC / precision-recall panels, which depict one "
            "trained model rather than a summary. Defaults to the lowest seed in the run, "
            "the fixed rule the manuscript declares. Metrics are computed independently "
            "per seed and summarised in the tables; they are never pooled across seeds."
        ),
    )
    parser.add_argument("--no-tsne", action="store_true")
    parser.add_argument(
        "--tsne-seed-index",
        type=int,
        default=0,
        help="Which training seed's checkpoints to embed, so each held-out window appears once.",
    )
    parser.add_argument("--tsne-max-samples", type=int, default=3000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    return parser


def load_ledger(run_dir: Path, task_id: str) -> tuple[pd.DataFrame, list[int]]:
    frame = pd.read_parquet(run_dir / "predictions.parquet")
    frame = frame[frame["task_id"] == task_id].reset_index(drop=True)
    if frame.empty:
        raise SystemExit(f"no {task_id} predictions in {run_dir}")
    return frame, sorted(int(seed) for seed in frame["seed"].unique())


def _labels(names: Sequence[str]) -> list[str]:
    return [CLASS_LABELS.get(name, name) for name in names]


# ------------------------------------------------------------------ confusion ---


def figure_confusion(frame: pd.DataFrame, out: Path) -> None:
    """Counts and row-normalised recall for one seed's held-out windows."""
    FigureStyle.apply()
    ledger = PredictionLedger(frame=frame, class_names=CLASS_IDS)
    matrices = confusion_matrices(ledger.y_true, ledger.y_pred, len(CLASS_IDS))
    raw, normalized = matrices["raw"], matrices["row_normalized"]
    labels = _labels(CLASS_IDS)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5))
    for position, (ax, matrix, title, fmt) in enumerate(
        ((axes[0], raw, "(a) Counts", "d"), (axes[1], normalized, "(b) Recall", ".2f"))
    ):
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels if position == 0 else [])
        ax.set_xlabel("Predicted")
        if position == 0:
            ax.set_ylabel("True")
        ax.set_title(title, loc="left")
        ax.grid(False)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    format(matrix[i, j], fmt) if fmt == ".2f" else f"{matrix[i, j]:,}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if normalized[i, j] > 0.55 else "black",
                )
    fig.colorbar(image, ax=axes, fraction=0.024, pad=0.02, label="Recall")
    save_figure(fig, out, "confmatrx_5class")


# ---------------------------------------------------------------------- curves ---


def figure_curves(frame: pd.DataFrame, out: Path) -> None:
    """One-vs-rest ROC and precision-recall, one figure each."""
    FigureStyle.apply()
    curves = curve_points(PredictionLedger(frame=frame, class_names=CLASS_IDS))

    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    for name in CLASS_IDS:
        entry = curves.get(name, {})
        if not entry.get("available"):
            continue
        ax.plot(
            entry["fpr"],
            entry["tpr"],
            color=CLASS_COLORS[name],
            linewidth=1.5,
            label=f"{CLASS_LABELS[name]} ({entry['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=0.9, label="Chance (0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right", fontsize=7.5, title="Class (AUC)", title_fontsize=7.5)
    save_figure(fig, out, "five_class_roc_curves")

    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    for name in CLASS_IDS:
        entry = curves.get(name, {})
        if not entry.get("available"):
            continue
        ax.plot(
            entry["recall"],
            entry["precision"],
            color=CLASS_COLORS[name],
            linewidth=1.5,
            label=f"{CLASS_LABELS[name]} ({entry['average_precision']:.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower left", fontsize=7.5, title="Class (avg. precision)", title_fontsize=7.5)
    save_figure(fig, out, "five_class_pr_curves")


# ------------------------------------------------------------- per participant ---


def figure_per_participant(frame: pd.DataFrame, seeds: list[int], out: Path) -> None:
    """Macro-F1 per held-out participant: bar = seed mean, marker = individual seed."""
    FigureStyle.apply()
    per_seed: dict[int, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for seed in seeds:
        subset = frame[frame["seed"] == seed].reset_index(drop=True)
        summary = subject_level_summary(PredictionLedger(frame=subset, class_names=CLASS_IDS))
        per_seed[seed] = {
            subject: entry["macro_f1"] for subject, entry in summary["per_subject"].items()
        }
        counts = {subject: entry["n_samples"] for subject, entry in summary["per_subject"].items()}

    subjects = sorted(counts)
    means = [float(np.mean([per_seed[s][sub] for s in seeds])) for sub in subjects]
    positions = np.arange(len(subjects), dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(
        positions,
        means,
        color=[FigureStyle.color(i) for i in range(len(subjects))],
        width=0.6,
        alpha=0.85,
    )
    for index, subject in enumerate(subjects):
        values = [per_seed[seed][subject] for seed in seeds]
        ax.plot(
            np.full(len(values), positions[index]),
            values,
            "o",
            markersize=3.5,
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=0.8,
            zorder=3,
        )
        ax.text(
            positions[index], max(values) + 0.03, f"{means[index]:.3f}", ha="center", fontsize=8
        )
        ax.text(
            positions[index], 0.03, f"n={counts[subject]:,}", ha="center", fontsize=7, color="white"
        )
    grand = float(np.mean(means))
    ax.axhline(grand, color="#333333", linestyle="--", linewidth=1.0)
    ax.annotate(
        f"mean {grand:.3f}",
        xy=(1.0, grand),
        xycoords=("axes fraction", "data"),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color="#333333",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"P{index + 1}" for index in range(len(subjects))])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Held-out participant")
    ax.set_ylabel("Macro F1")
    save_figure(fig, out, "five_class_per_participant")


# ----------------------------------------------------------------- sample flow ---


def figure_sample_flow(frame: pd.DataFrame, out: Path) -> None:
    """Analysable windows per participant and class, after every exclusion."""
    FigureStyle.apply()
    windows = frame.drop_duplicates("sample_id")
    subjects = sorted(windows["subject_id"].unique())
    counts = {
        name: [
            int(((windows["subject_id"] == subject) & (windows["true_class"] == name)).sum())
            for subject in subjects
        ]
        for name in CLASS_IDS
    }

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    positions = np.arange(len(subjects), dtype=float)
    bottom = np.zeros(len(subjects))
    for name in CLASS_IDS:
        values = np.array(counts[name], dtype=float)
        ax.bar(
            positions,
            values,
            bottom=bottom,
            width=0.62,
            color=CLASS_COLORS[name],
            label=CLASS_LABELS[name],
        )
        bottom += values
    for index, total in enumerate(bottom):
        ax.text(positions[index], total + 25, f"{int(total):,}", ha="center", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"P{index + 1}" for index in range(len(subjects))])
    ax.set_xlabel("Participant")
    ax.set_ylabel("Analysable windows")
    ax.set_ylim(0, bottom.max() * 1.12)
    ax.legend(loc="upper right", fontsize=7.5, ncols=2)
    save_figure(fig, out, "sample_flow")


# ------------------------------------------------------------ training curves ---


def figure_training_curves(outcomes: Sequence[dict[str, Any]], seeds: list[int], out: Path) -> None:
    """Refit loss and training-fit macro-F1 for every fold-seed combination."""
    FigureStyle.apply()
    subjects = sorted({str(entry["test_subject"]) for entry in outcomes})
    colors = {subject: FigureStyle.color(index) for index, subject in enumerate(subjects)}

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    for entry in outcomes:
        history = entry.get("training_history") or []
        if not history:
            continue
        epochs = [point["epoch"] for point in history]
        subject = str(entry["test_subject"])
        style = SEED_STYLES[seeds.index(int(entry["seed"])) % len(SEED_STYLES)]
        axes[0].plot(
            epochs,
            [point["train_loss"] for point in history],
            style,
            color=colors[subject],
            linewidth=1.2,
        )
        axes[1].plot(
            epochs,
            [point["val_macro_f1"] for point in history],
            style,
            color=colors[subject],
            linewidth=1.2,
        )
    axes[0].set_title("(a) Refit training loss", loc="left")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[1].set_title("(b) Refit training-fit macro F1", loc="left")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1 (training fit)")

    handles = [
        plt.Line2D([], [], color=colors[subject], linewidth=1.4, label=f"Held out P{index + 1}")
        for index, subject in enumerate(subjects)
    ] + [
        plt.Line2D(
            [], [], color="#555555", linestyle=SEED_STYLES[i], linewidth=1.2, label=f"Seed {seed}"
        )
        for i, seed in enumerate(seeds)
    ]
    axes[1].legend(handles=handles, loc="lower right", fontsize=7, ncols=2)
    fig.tight_layout()
    save_figure(fig, out, "training_curves_5class")


# --------------------------------------------------------------------- t-SNE ---


def compute_embeddings(
    run_dir: Path, task_id: str, data_root: Path, *, seed: int, max_samples: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Recompute held-out fusion embeddings from one seed's fold checkpoints.

    Restricting to a single seed keeps every held-out window in the projection exactly
    once; pooling seeds would plot each window three times.
    """
    import torch

    from bruxism.config import load_experiment_config
    from bruxism.data.dataset import WindowDataset
    from bruxism.data.labels import get_task
    from bruxism.models.baselines import build_neural_model
    from bruxism.preprocessing.normalization import Normalizer
    from bruxism.runner import prepare_data

    config = load_experiment_config(run_dir / "resolved_config.yaml")
    _, window_index, cache = prepare_data(config, data_root=data_root)
    task = get_task(task_id)
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    predictions = predictions[(predictions["task_id"] == task_id) & (predictions["seed"] == seed)]

    checkpoints = sorted((run_dir / "checkpoints").glob(f"{task_id}_*_seed{seed}.pt"))
    if not checkpoints:
        raise SystemExit(f"no seed-{seed} checkpoints under {run_dir / 'checkpoints'}")

    chunks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        architecture = payload["architecture"]["config"]
        model = build_neural_model(
            architecture.get("model_id", config.model_id),
            num_classes=task.num_classes,
            modality=architecture.get("modality", config.modality),
            emg_channels=cache.n_emg_channels,
            window_samples=config.data.window_samples(),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()

        normalizer = Normalizer.from_dict(payload["normalizer"])
        held_out = sorted(set(predictions["subject_id"]) - set(normalizer.fitted_on))
        sample_ids = list(predictions[predictions["subject_id"].isin(held_out)]["sample_id"])
        if not sample_ids:
            continue
        dataset = WindowDataset(
            window_index,
            sample_ids,
            task,
            cache,
            stage="test",
            normalizer=normalizer,
            modality=architecture.get("modality", config.modality),
        )
        with torch.no_grad():
            for start in range(0, len(dataset), 256):
                batch = [dataset[i] for i in range(start, min(start + 256, len(dataset)))]
                emg = torch.stack([item[0] for item in batch])
                mic = torch.stack([item[1] for item in batch])
                chunks.append(model.embed(emg, mic).numpy())
                labels.append(np.array([int(item[2]) for item in batch]))

    stacked = np.concatenate(chunks, axis=0)
    label_array = np.concatenate(labels, axis=0)
    if stacked.shape[0] > max_samples:
        keep = np.linspace(0, stacked.shape[0] - 1, max_samples).astype(int)
        stacked, label_array = stacked[keep], label_array[keep]
    return stacked, label_array, list(task.class_names)


def figure_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    out: Path,
    *,
    perplexity: float,
) -> None:
    from sklearn.manifold import TSNE

    FigureStyle.apply()
    projection = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=0,
    ).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for index, name in enumerate(class_names):
        mask = labels == index
        if not mask.any():
            continue
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=7,
            alpha=0.65,
            linewidths=0,
            color=CLASS_COLORS.get(name, FigureStyle.color(index)),
            label=f"{CLASS_LABELS.get(name, name)} (n={int(mask.sum()):,})",
        )
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(loc="best", fontsize=7.5, markerscale=2.0)
    save_figure(fig, out, "tsne_5class")


# ----------------------------------------------------------------------- main ---


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    frame, seeds = load_ledger(run_dir, args.task_id)
    outcomes = json.loads((run_dir / "selection" / "fold_outcomes.json").read_text())
    outcomes = [entry for entry in outcomes if entry["condition"]["task_id"] == args.task_id]

    figure_seed = seeds[0] if args.figure_seed is None else args.figure_seed
    if figure_seed not in seeds:
        raise SystemExit(f"seed {figure_seed} is not in this run (have {seeds})")
    single = frame[frame["seed"] == figure_seed].reset_index(drop=True)

    figure_confusion(single, out)
    figure_curves(single, out)
    figure_per_participant(frame, seeds, out)
    figure_sample_flow(frame, out)
    figure_training_curves(outcomes, seeds, out)
    written = [
        "confmatrx_5class",
        "five_class_roc_curves",
        "five_class_pr_curves",
        "five_class_per_participant",
        "sample_flow",
        "training_curves_5class",
    ]

    if not args.no_tsne:
        if args.data_root is None:
            print("skipping t-SNE: --data-root is required to recompute held-out embeddings")
        else:
            embeddings, labels, class_names = compute_embeddings(
                run_dir,
                args.task_id,
                args.data_root.resolve(),
                seed=seeds[args.tsne_seed_index],
                max_samples=args.tsne_max_samples,
            )
            figure_tsne(embeddings, labels, class_names, out, perplexity=args.tsne_perplexity)
            written.append("tsne_5class")

    bundle = json.loads((run_dir / "run_bundle.json").read_text())
    provenance = {
        "run_id": bundle["run_id"],
        "config_hash": bundle["config_hash"],
        "manifest_hash": bundle["manifest_hash"],
        "window_index_hash": bundle["window_index_hash"],
        "seeds": seeds,
        "n_predictions": int(len(frame)),
        "figure_seed": figure_seed,
        "figures": written,
        "note": (
            "The confusion, ROC and precision-recall panels depict one trained model per "
            "fold at the lowest seed index -- a fixed rule, not a choice made after seeing "
            "the scores -- because a curve drawn over pooled seeds would mix models. The "
            "per-participant figure shows all seeds: bar heights are seed means and the "
            "markers are the individual seeds. The t-SNE uses one seed's checkpoints so "
            "each held-out window appears exactly once."
        ),
    }
    (out / "manuscript_figures_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(written)} manuscript figure(s) (PNG + PDF) to {out}")
    for stem in written:
        print(f"  {stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
