"""Run diagnostics: what the model learned, how it was selected, and where it failed.

Every function here reads a saved artifact of one run -- the prediction ledger, the metrics
summary derived from it, or the per-fold outcome records -- and nothing else. None of them
re-runs a model or accepts a hand-entered number, with one declared exception: the embedding
projection recomputes held-out representations from a saved checkpoint, because an embedding
is not stored in the ledger. That function records the checkpoints it used.

The figures split into three groups:

selection and training
    :func:`plot_training_curves_by_seed`, :func:`plot_hyperparameter_selection` -- how the
    epoch budget and hyperparameters were chosen on inner folds only.
performance
    :func:`plot_per_class_performance`, :func:`plot_participant_class_recall`,
    :func:`plot_seed_stability` -- where the held-out performance actually comes from.
trust
    :func:`plot_calibration`, :func:`plot_error_timeline`, :func:`plot_embedding_projection`
    -- whether the probabilities mean anything, when errors happen, and what the learned
    space looks like.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bruxism.evaluation.metrics import PredictionLedger  # noqa: E402
from bruxism.utils.io import write_csv  # noqa: E402
from bruxism.utils.logging import get_logger  # noqa: E402
from bruxism.visualization.paper_figures import FigureStyle, caveat, save_figure  # noqa: E402

__all__ = [
    "expected_calibration_error",
    "plot_calibration",
    "plot_embedding_projection",
    "plot_error_timeline",
    "plot_hyperparameter_selection",
    "plot_participant_class_recall",
    "plot_per_class_performance",
    "plot_run_scorecard",
    "plot_seed_stability",
    "plot_training_curves_by_seed",
]

logger = get_logger(__name__)


def _pretty(name: str) -> str:
    return str(name).replace("_", " ")


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if np.isnan(number) else format(number, spec)


# ------------------------------------------------------------------ scorecard ---


def plot_run_scorecard(
    *,
    run_id: str,
    config: dict[str, Any],
    bundle: dict[str, Any],
    source_state: dict[str, Any],
    environment: dict[str, Any],
    metrics: dict[str, Any],
    task_id: str,
    n_folds: int,
    output_dir: Path,
    stem: str = "00_run_scorecard",
) -> list[Path]:
    """A cover page for the run: identity, protocol, headline numbers and their caveats.

    Everything on it is read from the run bundle, so the page cannot claim a number the
    ledger does not contain. It exists so that a figure folder copied into a slide deck or
    a supplement still says which run, which data and which commit produced it.
    """
    FigureStyle.apply()
    conditions = [
        entry for entry in metrics.get("conditions", {}).values() if entry.get("task_id") == task_id
    ]
    seeds = sorted({int(entry.get("seed", 0)) for entry in conditions})

    def across_seeds(metric: str) -> tuple[float | None, float | None]:
        values = [
            entry["subject_level"][metric]["mean"]
            for entry in conditions
            if isinstance(entry.get("subject_level", {}).get(metric), dict)
            and entry["subject_level"][metric] is not None
        ]
        if not values:
            return None, None
        return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)

    headline = [
        ("Accuracy", *across_seeds("accuracy")),
        ("Balanced accuracy", *across_seeds("balanced_accuracy")),
        ("Macro F1", *across_seeds("macro_f1")),
        ("Macro one-vs-rest AUC", *across_seeds("macro_roc_auc")),
    ]
    n_windows = sum(
        entry.get("pooled_windows", {}).get("n_samples", 0)
        for entry in conditions
        if int(entry.get("seed", 0)) == (seeds[0] if seeds else 0)
    )
    data = config.get("data", {})
    training = config.get("training", {})
    selection = config.get("selection", {})

    identity = [
        ("run id", run_id),
        (
            "task / model / modality",
            f"{task_id} / {config.get('model_id')} / {config.get('modality')}",
        ),
        (
            "source commit",
            f"{str(source_state.get('commit') or 'unknown')[:12]}"
            f"{'  (DIRTY TREE)' if source_state.get('is_dirty') else ''}",
        ),
        ("config hash", bundle.get("config_hash", "unknown")),
        ("data manifest hash", bundle.get("manifest_hash", "unknown")),
        ("window index hash", bundle.get("window_index_hash", "unknown")),
        (
            "torch / python",
            f"{environment.get('torch', {}).get('version', '?')} / "
            f"{str(environment.get('python', '?')).split()[0]}",
        ),
    ]
    protocol = [
        (
            "segmentation",
            f"{data.get('segmentation_policy')}, "
            f"{data.get('window_seconds')}s window / {data.get('stride_seconds')}s "
            f"stride, {data.get('guard_seconds')}s guard",
        ),
        ("held-out folds executed", f"{n_folds} (leave-one-subject-out, nested inner selection)"),
        ("seeds", ", ".join(str(seed) for seed in seeds) or "n/a"),
        (
            "selection objective",
            f"{selection.get('objective')} "
            f"(epoch budget rule: {selection.get('epoch_budget_rule')}, "
            f"max {selection.get('max_epochs')} epochs)",
        ),
        (
            "loss / class weights",
            f"{training.get('loss')} (gamma={training.get('focal_gamma')}) / "
            f"{training.get('class_weight_scheme')}",
        ),
        ("held-out windows scored", f"{n_windows:,} per seed"),
    ]

    fig = plt.figure(figsize=(11.0, 7.6))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis("off")

    ax.text(0.5, 0.965, "Run scorecard", ha="center", fontsize=17, fontweight="bold")
    ax.text(
        0.5,
        0.932,
        "Instructed awake jaw / tooth-contact task classification from surface EMG and audio",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )

    y = 0.885
    ax.text(0.045, y, "Identity and provenance", fontsize=10.5, fontweight="bold")
    y -= 0.030
    for key, value in identity:
        ax.text(0.055, y, f"{key}", fontsize=8.5, color="#555555")
        ax.text(0.315, y, str(value), fontsize=8.5, family="monospace")
        y -= 0.0265

    y -= 0.018
    ax.text(0.045, y, "Protocol", fontsize=10.5, fontweight="bold")
    y -= 0.030
    for key, value in protocol:
        ax.text(0.055, y, f"{key}", fontsize=8.5, color="#555555")
        ax.text(0.315, y, str(value), fontsize=8.5)
        y -= 0.0265

    y -= 0.020
    ax.text(
        0.045, y, "Held-out results (participant-level, primary)", fontsize=10.5, fontweight="bold"
    )
    y -= 0.012
    box_y = y - 0.135
    ax.add_patch(
        plt.Rectangle(
            (0.045, box_y),
            0.91,
            0.128,
            transform=ax.transAxes,
            facecolor="#F3F6F9",
            edgecolor="#CCD4DC",
            linewidth=0.8,
        )
    )
    for index, (name, mean, std) in enumerate(headline):
        x = 0.075 + index * 0.225
        ax.text(x, box_y + 0.085, name, fontsize=8.5, color="#555555")
        ax.text(
            x,
            box_y + 0.036,
            _fmt(mean),
            fontsize=19,
            fontweight="bold",
            color=FigureStyle.color(index),
        )
        ax.text(
            x,
            box_y + 0.012,
            f"+/- {_fmt(std)} across {len(seeds)} seed(s)",
            fontsize=7.5,
            color="#666666",
        )

    ax.text(
        0.5,
        0.075,
        "Each headline value is the mean across the five held-out participants of a metric "
        "computed within each participant, then averaged over seeds. Participants, not "
        "windows, are the unit of generalisation.",
        ha="center",
        fontsize=8,
        color="#333333",
        wrap=True,
    )
    ax.text(
        0.5,
        0.035,
        "SCOPE: instructed, awake, laboratory jaw tasks from five participants with a prior "
        "bruxism diagnosis. Not a clinical bruxism detector, not sleep bruxism, not validated "
        "on spontaneous behaviour. 'natural_bruxing' is an acquisition filename token and is "
        "reported as instructed grinding.",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#8A3D00",
        wrap=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in (".png", ".pdf"):
        path = output_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches=None)
        written.append(path)
    plt.close(fig)
    return written


# --------------------------------------------------------- training / selection ---


def plot_training_curves_by_seed(
    fold_outcomes: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    stem: str = "11_training_curves",
) -> list[Path]:
    """Per-epoch loss and objective of every refit, one column per seed.

    Splitting by seed keeps the lines readable when a run has several seeds x five folds,
    and it makes seed-to-seed instability visible as a difference between columns rather
    than as noise inside one crowded panel.
    """
    FigureStyle.apply()
    usable = [outcome for outcome in fold_outcomes if outcome.get("training_history")]
    if not usable:
        raise ValueError("no fold recorded a training history")

    seeds = sorted({int(outcome.get("seed", 0)) for outcome in usable})
    fig, axes = plt.subplots(
        2,
        len(seeds),
        figsize=(4.6 * len(seeds) + 1.0, 7.0),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    for column, seed in enumerate(seeds):
        for outcome in [o for o in usable if int(o.get("seed", 0)) == seed]:
            history = outcome["training_history"]
            epochs = [record["epoch"] for record in history]
            fold = int(outcome.get("outer_fold", 0))
            colour = FigureStyle.color(fold)
            label = f"fold {fold} (held out {outcome.get('test_subject')})"
            axes[0][column].plot(
                epochs,
                [r["train_loss"] for r in history],
                color=colour,
                linewidth=1.4,
                label=label,
            )
            axes[1][column].plot(
                epochs,
                [r["val_macro_f1"] for r in history],
                color=colour,
                linewidth=1.4,
                label=label,
            )
            budget = outcome.get("epoch_budget")
            if budget:
                axes[0][column].axvline(
                    budget, color=colour, linewidth=0.7, linestyle=":", alpha=0.6
                )
        axes[0][column].set_title(f"seed {seed}")
        axes[1][column].set_xlabel("Epoch")
    axes[0][0].set_ylabel("Training loss")
    axes[1][0].set_ylabel("Macro F1 (fit diagnostic)")
    axes[0][0].legend(fontsize=6.5)

    scopes = sorted({str(outcome.get("history_scope", "unknown")) for outcome in usable})
    fig.suptitle(
        "Final-refit training curves per outer fold (dotted line: that fold's epoch budget)",
        y=0.995,
    )
    caveat(
        fig,
        "The held-out participant contributes to no curve here. The refit runs for exactly "
        "the epoch budget chosen on inner folds, with no early stopping and no validation "
        f"set, so the per-epoch numbers (scope: {', '.join(scopes)}) are measured on the "
        "training data and are fit diagnostics, not validation. Inner-fold validation curves "
        "are what selected the budget; see the hyperparameter-selection figure.",
    )
    return save_figure(fig, output_dir, stem)


def plot_hyperparameter_selection(
    fold_outcomes: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    stem: str = "12_hyperparameter_selection",
) -> list[Path]:
    """What the inner search scored and what it therefore selected, per outer fold.

    Left: each trial's mean inner-validation objective with its spread across inner folds,
    with the selected trial marked. Right: the epoch budget derived from the winning trial's
    inner best epochs. Together these are the reproducibility evidence for the
    hyperparameters and epoch count reported in the manuscript.
    """
    FigureStyle.apply()
    rows: list[dict[str, Any]] = []
    for outcome in fold_outcomes:
        selected = outcome.get("selected_hyperparameters") or {}
        for trial in outcome.get("inner_trials", []):
            values = [
                entry.get("best_value")
                for entry in trial.get("inner_selections", [])
                if entry.get("best_value") is not None
            ]
            rows.append(
                {
                    "fold": int(outcome.get("outer_fold", 0)),
                    "seed": int(outcome.get("seed", 0)),
                    "test_subject": outcome.get("test_subject"),
                    "trial_id": trial.get("trial_id"),
                    "settings": ", ".join(
                        f"{k}={v}" for k, v in sorted((trial.get("hyperparameters") or {}).items())
                    )
                    or "config defaults",
                    "mean": trial.get("mean_objective"),
                    "std": trial.get("std_objective"),
                    "n_inner": len(values),
                    "failed": bool(trial.get("failed")),
                    "selected": (trial.get("hyperparameters") or {}) == selected,
                    "epoch_budget": outcome.get("epoch_budget"),
                    "objective": (
                        trial.get("inner_selections", [{}])[0].get("objective", "objective")
                        if trial.get("inner_selections")
                        else "objective"
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no inner-search trials were recorded (was a search space configured?)")

    objective = str(frame["objective"].iloc[0])
    settings = list(dict.fromkeys(frame["settings"]))
    folds = sorted(frame["fold"].unique())
    seeds = sorted(frame["seed"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), width_ratios=[1.9, 1.0])
    width = 0.8 / max(len(settings), 1)
    for index, setting in enumerate(settings):
        subset = frame[frame["settings"] == setting]
        grouped = subset.groupby("fold", observed=True).agg(
            mean=("mean", "mean"), std=("std", "mean")
        )
        positions = np.array([folds.index(f) for f in grouped.index], dtype=float)
        positions = positions + index * width - 0.4 + width / 2
        axes[0].bar(
            positions,
            grouped["mean"].to_numpy(dtype=float),
            yerr=grouped["std"].to_numpy(dtype=float),
            width=width * 0.9,
            color=FigureStyle.color(index),
            label=setting,
            error_kw={"linewidth": 0.7, "ecolor": "#555555"},
        )
    # One marker per (fold, trial) any seed selected, placed clear of the error bar so it
    # cannot be read as a data point. With several seeds the label says how many chose it.
    chosen = (
        frame[frame["selected"]]
        .groupby(["fold", "settings"], observed=True)
        .agg(value=("mean", "max"), spread=("std", "max"), n_seeds=("seed", "nunique"))
        .reset_index()
    )
    ceiling = float(np.nanmax(frame["mean"].to_numpy(dtype=float))) if len(frame) else 1.0
    for row in chosen.to_dict("records"):
        value = row["value"]
        if value is None or np.isnan(float(value)):
            continue
        index = settings.index(row["settings"])
        position = folds.index(row["fold"]) + index * width - 0.4 + width / 2
        top = float(value) + float(row["spread"] or 0.0) + 0.02
        ceiling = max(ceiling, top)
        axes[0].plot([position], [top], marker="v", markersize=6, color="#000000", linestyle="none")
        if len(seeds) > 1:
            axes[0].annotate(
                f"{int(row['n_seeds'])}/{len(seeds)}",
                (position, top),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=6,
                color="#333333",
            )
    axes[0].set_ylim(0, ceiling * 1.16)
    axes[0].set_xticks(np.arange(len(folds)))
    axes[0].set_xticklabels(
        [
            f"fold {fold}\n(held out {frame[frame['fold'] == fold]['test_subject'].iloc[0]})"
            for fold in folds
        ],
        fontsize=8,
    )
    axes[0].set_ylabel(f"Mean inner-validation {_pretty(objective)}")
    axes[0].set_title(
        f"Inner hyperparameter search ({len(settings)} trial(s) x inner folds"
        + (f", averaged over {len(seeds)} seeds)" if len(seeds) > 1 else ")")
    )
    axes[0].legend(fontsize=7, title="trial", title_fontsize=7)

    budgets = (
        frame.drop_duplicates(["fold", "seed"])
        .groupby("fold", observed=True)["epoch_budget"]
        .apply(list)
    )
    for offset, seed in enumerate(seeds):
        values = [
            frame[(frame["fold"] == fold) & (frame["seed"] == seed)]["epoch_budget"].iloc[0]
            if not frame[(frame["fold"] == fold) & (frame["seed"] == seed)].empty
            else np.nan
            for fold in folds
        ]
        axes[1].plot(
            np.arange(len(folds)),
            values,
            marker="o",
            markersize=5,
            linewidth=1.2,
            color=FigureStyle.color(offset),
            label=f"seed {seed}",
        )
    axes[1].set_xticks(np.arange(len(folds)))
    axes[1].set_xticklabels([f"fold {fold}" for fold in folds], fontsize=8)
    axes[1].set_ylabel("Epochs")
    axes[1].set_title("Refit epoch budget selected per fold")
    axes[1].legend(fontsize=7)
    axes[1].set_ylim(0, max(1.0, float(np.nanmax([v for vs in budgets for v in vs])) * 1.25))

    caveat(
        fig,
        "Every value here was measured on inner-validation participants only; the outer "
        "held-out participant is unreachable during selection. The marker under a bar is "
        "the trial the prespecified rule selected (highest mean inner objective, then lower "
        "across-fold spread, then lexicographic trial id). The epoch budget is the rounded-up "
        "median of the winning trial's inner best epochs, clamped to the configured range.",
    )
    return save_figure(fig, output_dir, stem)


# ---------------------------------------------------------------- performance ---


def plot_per_class_performance(
    metrics_entry: dict[str, Any],
    output_dir: Path,
    *,
    stem: str = "17_per_class_performance",
) -> list[Path]:
    """Precision, recall, F1 and one-vs-rest AUC for every class, with support annotated."""
    FigureStyle.apply()
    pooled = metrics_entry.get("pooled_windows", {})
    per_class = pooled.get("per_class") or {}
    if not per_class:
        raise ValueError("metrics contain no per-class block")
    auc = pooled.get("roc_auc_per_class") or {}
    average_precision = pooled.get("average_precision_per_class") or {}

    # Label order, not dictionary order: the metrics JSON is written with sorted keys, so
    # reading `per_class` directly would put the classes in a different order from every
    # other figure, including the confusion matrix.
    declared = pooled.get("class_names") or metrics_entry.get("class_names") or list(per_class)
    names = [name for name in declared if name in per_class]
    metrics_shown = ("precision", "recall", "f1")
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 4.5), width_ratios=[1.6, 1.0], gridspec_kw={"wspace": 0.42}
    )
    width = 0.8 / len(metrics_shown)
    for index, metric in enumerate(metrics_shown):
        positions = np.arange(len(names)) + index * width - 0.4 + width / 2
        values = [float(per_class[name].get(metric, 0.0)) for name in names]
        axes[0].bar(
            positions,
            values,
            width=width * 0.9,
            color=FigureStyle.color(index),
            label=_pretty(metric),
        )
        for position, value in zip(positions, values):
            axes[0].text(
                position, value + 0.015, f"{value:.2f}", ha="center", fontsize=6.5, rotation=90
            )
    axes[0].set_xticks(np.arange(len(names)))
    axes[0].set_xticklabels(
        [f"{_pretty(name)}\n(n={per_class[name].get('support', 0):,})" for name in names],
        fontsize=8,
    )
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Per-class held-out performance (pooled windows)")
    axes[0].legend(fontsize=7, ncols=3)

    for index, (label, source) in enumerate(
        (("one-vs-rest ROC AUC", auc), ("average precision", average_precision))
    ):
        positions = np.arange(len(names)) + index * 0.4 - 0.2
        scores = [source.get(name) for name in names]
        axes[1].barh(
            positions,
            [0.0 if score is None else float(score) for score in scores],
            height=0.36,
            color=FigureStyle.color(index + 3),
            label=label,
        )
        for position, score in zip(positions, scores):
            axes[1].text(
                0.015,
                float(position),
                _fmt(score),
                va="center",
                fontsize=7,
                color="white" if (score or 0) > 0.25 else "black",
            )
    axes[1].set_yticks(np.arange(len(names)))
    axes[1].set_yticklabels([_pretty(name) for name in names], fontsize=8)
    axes[1].invert_yaxis()
    # Headroom to the right of the longest bar, so the legend never lands on one.
    axes[1].set_xlim(0, 1.28)
    axes[1].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axes[1].set_xlabel("Score")
    axes[1].set_title("Ranking quality per class")
    axes[1].legend(fontsize=7, loc="lower right")

    caveat(
        fig,
        "Computed from held-out probabilities pooled over folds, never from hard labels or "
        "from a confusion matrix. Pooled windows are descriptive: they are correlated within "
        "a participant and recording, so no confidence interval over windows would be valid. "
        "A class's no-skill average precision equals its prevalence, not 0.5.",
    )
    return save_figure(fig, output_dir, stem)


def plot_participant_class_recall(
    ledger: PredictionLedger,
    output_dir: Path,
    *,
    stem: str = "18_participant_class_recall",
) -> list[Path]:
    """Recall for every (held-out participant, class) cell, with support in the cell.

    A mean macro-F1 hides which participant failed and on which class. This shows it: an
    empty or near-zero row is a participant the model did not transfer to, and a near-zero
    column is a class no fold learned.
    """
    FigureStyle.apply()
    frame = ledger.frame
    if frame.empty:
        raise ValueError("empty prediction ledger")
    subjects = sorted(frame["subject_id"].unique())
    names = list(ledger.class_names)

    recall = np.full((len(subjects), len(names)), np.nan)
    support = np.zeros((len(subjects), len(names)), dtype=int)
    for row, subject in enumerate(subjects):
        subset = frame[frame["subject_id"] == subject]
        for column, _name in enumerate(names):
            truth = subset[subset["true_label"] == column]
            support[row, column] = len(truth)
            if len(truth):
                recall[row, column] = float((truth["predicted_label"] == column).mean())

    fig, ax = plt.subplots(figsize=(1.55 * len(names) + 3.0, 0.62 * len(subjects) + 3.0))
    image = ax.imshow(recall, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([_pretty(name) for name in names], rotation=35, ha="right")
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects)
    ax.set_xlabel("True class")
    ax.set_ylabel("Held-out participant")
    ax.grid(False)
    for row in range(len(subjects)):
        for column in range(len(names)):
            value = recall[row, column]
            text = (
                "no windows"
                if support[row, column] == 0
                else (f"{value:.2f}\nn={support[row, column]:,}")
            )
            ax.text(column, row, text, ha="center", va="center", fontsize=7.5, color="#111111")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Recall")
    ax.set_title("Per-participant, per-class recall on held-out folds")
    caveat(
        fig,
        "Each row is that participant's own leave-one-subject-out fold, so no cell contains "
        "a training prediction. Cells are recall (true positives / that participant's windows "
        "of the class); a grey cell means the participant contributed no window of that class "
        "and the metric is undefined rather than zero.",
    )
    return save_figure(fig, output_dir, stem)


def plot_seed_stability(
    metrics: dict[str, Any],
    task_id: str,
    output_dir: Path,
    *,
    stem: str = "20_seed_stability",
    metric: str = "macro_f1",
) -> list[Path]:
    """How much the result moves when only the random seed changes.

    Seeds are summarised, never selected between. If the spread across seeds is comparable
    to the spread across participants, a single-seed number would be reporting noise.
    """
    FigureStyle.apply()
    conditions = [
        entry for entry in metrics.get("conditions", {}).values() if entry.get("task_id") == task_id
    ]
    if not conditions:
        raise ValueError(f"metrics contain no condition for task {task_id!r}")
    conditions.sort(key=lambda entry: int(entry.get("seed", 0)))
    if len(conditions) < 2:
        raise ValueError("only one seed was run; a seed-stability figure needs at least two")

    subjects = sorted(
        {
            subject
            for entry in conditions
            for subject in entry.get("subject_level", {}).get("per_subject", {})
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), width_ratios=[1.5, 1.0])

    width = 0.8 / len(conditions)
    for index, entry in enumerate(conditions):
        per_subject = entry.get("subject_level", {}).get("per_subject", {})
        values = [per_subject.get(subject, {}).get(metric) for subject in subjects]
        positions = np.arange(len(subjects)) + index * width - 0.4 + width / 2
        axes[0].bar(
            positions,
            [0.0 if value is None else float(value) for value in values],
            width=width * 0.9,
            color=FigureStyle.color(index),
            label=f"seed {entry.get('seed')}",
        )
    for position, subject in enumerate(subjects):
        values = [
            entry.get("subject_level", {}).get("per_subject", {}).get(subject, {}).get(metric)
            for entry in conditions
        ]
        usable = [float(value) for value in values if value is not None]
        if len(usable) > 1:
            axes[0].text(
                float(position),
                max(usable) + 0.03,
                f"range {max(usable) - min(usable):.3f}",
                ha="center",
                fontsize=7,
                color="#444444",
            )
    axes[0].set_xticks(np.arange(len(subjects)))
    axes[0].set_xticklabels(subjects)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel(_pretty(metric))
    axes[0].set_xlabel("Held-out participant")
    axes[0].set_title(f"Per-participant {_pretty(metric)} by seed")
    axes[0].legend(fontsize=7, ncols=len(conditions))

    tracked = ("accuracy", "balanced_accuracy", "macro_f1", "macro_roc_auc")
    for index, name in enumerate(tracked):
        values = [
            entry.get("subject_level", {}).get(name, {}).get("mean")
            if isinstance(entry.get("subject_level", {}).get(name), dict)
            else None
            for entry in conditions
        ]
        axes[1].plot(
            [int(entry.get("seed", 0)) for entry in conditions],
            [np.nan if value is None else float(value) for value in values],
            marker="o",
            markersize=5,
            linewidth=1.3,
            color=FigureStyle.color(index),
            label=_pretty(name),
        )
    axes[1].set_xticks([int(entry.get("seed", 0)) for entry in conditions])
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Participant-level mean")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Headline metrics per seed")
    axes[1].legend(fontsize=7)

    caveat(
        fig,
        "Metrics are computed independently per seed and then summarised; probabilities are "
        "not averaged across seeds and no seed is selected as best. A seed-to-seed range "
        "comparable to the participant-to-participant range means a single-seed number is "
        "not a stable estimate.",
    )
    return save_figure(fig, output_dir, stem)


# ---------------------------------------------------------------------- trust ---


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, *, bins: int = 10
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Equal-width-bin ECE plus the per-bin confidence, accuracy and count.

    Returns ``(ece, mean_confidence, accuracy, counts)`` with one entry per non-empty bin.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, bins - 1)
    mean_confidence, accuracy, counts = [], [], []
    total = float(confidence.size)
    ece = 0.0
    for position in range(bins):
        mask = index == position
        count = int(mask.sum())
        counts.append(count)
        if count == 0:
            mean_confidence.append(np.nan)
            accuracy.append(np.nan)
            continue
        bin_confidence = float(confidence[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        mean_confidence.append(bin_confidence)
        accuracy.append(bin_accuracy)
        ece += (count / total) * abs(bin_accuracy - bin_confidence)
    return ece, np.array(mean_confidence), np.array(accuracy), np.array(counts)


def plot_calibration(
    ledger: PredictionLedger,
    output_dir: Path,
    *,
    stem: str = "19_calibration",
    bins: int = 10,
) -> list[Path]:
    """Are the held-out probabilities trustworthy as probabilities?

    A reliability diagram plus the confidence histogram split by correctness. The manuscript
    reports AUC from these probabilities, so whether they are calibrated is a question a
    reviewer will ask; this answers it with a number (ECE) rather than an assertion.
    """
    FigureStyle.apply()
    frame = ledger.frame
    if frame.empty:
        raise ValueError("empty prediction ledger")
    scores = ledger.y_score
    confidence = scores.max(axis=1)
    correct = (ledger.y_pred == ledger.y_true).astype(np.float64)
    ece, bin_confidence, bin_accuracy, counts = expected_calibration_error(
        confidence, correct, bins=bins
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].plot(
        [0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.0, label="perfect calibration"
    )
    usable = ~np.isnan(bin_accuracy)
    axes[0].plot(
        bin_confidence[usable],
        bin_accuracy[usable],
        marker="o",
        markersize=5,
        linewidth=1.5,
        color=FigureStyle.color(0),
        label="observed",
    )
    for x, y, count in zip(bin_confidence[usable], bin_accuracy[usable], counts[usable]):
        axes[0].annotate(
            f"{count:,}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=6.5,
            color="#555555",
        )
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Predicted probability of the predicted class")
    axes[0].set_ylabel("Observed accuracy")
    axes[0].set_title(f"Reliability diagram (ECE = {ece:.3f}, {len(confidence):,} windows)")
    axes[0].legend(fontsize=7, loc="upper left")

    edges = np.linspace(0.0, 1.0, bins + 1)
    axes[1].hist(
        [confidence[correct == 1], confidence[correct == 0]],
        bins=edges,
        stacked=True,
        color=[FigureStyle.color(2), FigureStyle.color(4)],
        label=["correct", "incorrect"],
    )
    axes[1].set_xlabel("Predicted probability of the predicted class")
    axes[1].set_ylabel("Windows")
    axes[1].set_title("Confidence distribution")
    axes[1].legend(fontsize=7)

    chance = 1.0 / max(len(ledger.class_names), 1)
    axes[1].axvline(chance, color="#999999", linestyle=":", linewidth=1.0)
    axes[1].text(
        chance,
        axes[1].get_ylim()[1] * 0.96,
        f" chance {chance:.2f}",
        fontsize=6.5,
        color="#666666",
        va="top",
    )

    caveat(
        fig,
        "Softmax outputs of the evaluated checkpoints, uncalibrated -- no temperature "
        "scaling or isotonic fit was applied, and none could be fitted without spending a "
        "participant. A curve below the diagonal means the model is overconfident. Pooled "
        "over correlated windows, so this is descriptive.",
    )
    return save_figure(fig, output_dir, stem)


def plot_error_timeline(
    ledger: PredictionLedger,
    output_dir: Path,
    *,
    stem: str = "21_error_timeline",
    max_recordings: int = 8,
    subject_id: str | None = None,
) -> list[Path]:
    """Predictions in recording time, so a reader can see *when* the model is wrong.

    Errors that cluster at the start of a trial mean something different from errors spread
    uniformly: the first says the guard interval is too narrow or the participant took time
    to comply, the second says the classes genuinely overlap. A confusion matrix cannot tell
    them apart.
    """
    FigureStyle.apply()
    frame = ledger.frame
    if frame.empty:
        raise ValueError("empty prediction ledger")

    if subject_id is None:
        by_subject = (
            frame.assign(correct=frame["predicted_label"] == frame["true_label"])
            .groupby("subject_id", observed=True)["correct"]
            .mean()
            .sort_values()
        )
        # The median participant: neither the best-case nor the worst-case story.
        subject_id = str(by_subject.index[len(by_subject) // 2])
    subset = frame[frame["subject_id"] == subject_id]

    recordings = (
        subset.groupby("recording_id", observed=True)
        .size()
        .sort_values(ascending=False)
        .index.tolist()
    )
    # One recording per condition first, so every class is represented before repeats.
    by_condition: dict[str, str] = {}
    for recording in recordings:
        condition = str(subset[subset["recording_id"] == recording]["condition"].iloc[0])
        by_condition.setdefault(condition, recording)
    chosen = list(by_condition.values())[:max_recordings]
    if not chosen:
        raise ValueError(f"participant {subject_id} has no recordings in the ledger")

    names = list(ledger.class_names)
    # A shared time axis: every recording starts at its own t=0, so the rows are directly
    # comparable and only the bottom row needs tick labels.
    span = float(subset[subset["recording_id"].isin(chosen)]["end_seconds"].max())
    fig, axes = plt.subplots(
        len(chosen), 1, figsize=(12.0, 0.92 * len(chosen) + 2.0), squeeze=False, sharex=True
    )
    for row, recording in enumerate(chosen):
        ax = axes[row][0]
        block = subset[subset["recording_id"] == recording].sort_values("start_seconds")
        stride = float(
            np.median(np.diff(block["start_seconds"].to_numpy()))
            if len(block) > 1
            else block["end_seconds"].iloc[0] - block["start_seconds"].iloc[0]
        )
        for _, window in block.iterrows():
            start = float(window["start_seconds"])
            ax.add_patch(
                plt.Rectangle(
                    (start, 0.55),
                    stride,
                    0.4,
                    color=FigureStyle.color(int(window["true_label"])),
                    linewidth=0,
                )
            )
            ax.add_patch(
                plt.Rectangle(
                    (start, 0.05),
                    stride,
                    0.4,
                    color=FigureStyle.color(int(window["predicted_label"])),
                    linewidth=0,
                )
            )
            if int(window["predicted_label"]) != int(window["true_label"]):
                ax.plot(
                    [start + stride / 2],
                    [0.25],
                    marker="x",
                    markersize=3.2,
                    color="#000000",
                    linestyle="none",
                )
        ax.set_xlim(0.0, span)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.75])
        ax.set_yticklabels(["predicted", "true"], fontsize=7)
        ax.grid(False)
        accuracy = float((block["predicted_label"] == block["true_label"]).mean())
        ax.set_ylabel(
            f"{block['condition'].iloc[0]}\n{accuracy:.0%} correct",
            fontsize=7.5,
            rotation=0,
            ha="right",
            va="center",
            labelpad=42,
        )
        if row == len(chosen) - 1:
            ax.set_xlabel("Time within recording (s)")

    handles = [
        plt.Line2D([], [], color=FigureStyle.color(index), linewidth=7, label=_pretty(name))
        for index, name in enumerate(names)
    ]
    handles.append(
        plt.Line2D(
            [], [], color="#000000", marker="x", linestyle="none", markersize=4, label="error"
        )
    )
    axes[0][0].legend(
        handles=handles,
        fontsize=6.5,
        ncols=len(handles),
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
    )
    fig.suptitle(
        f"Held-out predictions in recording time -- participant {subject_id} "
        "(median accuracy of the cohort)",
        y=0.995,
    )
    caveat(
        fig,
        "Each block is one window drawn at the decision stride, so the tiling is contiguous "
        "even though the windows themselves overlap by half. Gaps are intervals the "
        "segmentation policy excluded: transition guards, the startup guard, and any "
        "trigger-low interval. Predictions come from the fold in which this participant was "
        "held out.",
    )
    return save_figure(fig, output_dir, stem)


def plot_embedding_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path,
    *,
    stem: str = "22_embedding_tsne",
    seed: int = 0,
    perplexity: float = 30.0,
    checkpoint_id: str = "",
    embedding_source: str = "fusion embedding",
) -> tuple[list[Path], dict[str, Any]]:
    """One t-SNE projection, coloured twice: by class, and by participant.

    The second panel is the honest one. If the held-out embeddings cluster by participant
    rather than by class, the network has learned who is wearing the electrodes as much as
    what they are doing -- which is exactly the failure mode a five-participant study is
    most exposed to, and which a class-coloured plot alone would hide.
    """
    from sklearn.manifold import TSNE

    FigureStyle.apply()
    matrix = np.asarray(embeddings, dtype=np.float64)
    n_samples = int(matrix.shape[0])
    if n_samples < 10:
        raise ValueError(f"only {n_samples} embeddings; too few to project")
    effective = float(min(perplexity, max(5.0, (n_samples - 1) / 3.0)))
    projected = TSNE(
        n_components=2,
        perplexity=effective,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(matrix)

    unique_subjects = sorted(set(map(str, subjects)))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    for index, name in enumerate(class_names):
        mask = labels == index
        if not mask.any():
            continue
        axes[0].scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=6,
            alpha=0.6,
            linewidths=0,
            color=FigureStyle.color(index),
            label=f"{_pretty(name)} (n={int(mask.sum())})",
        )
    for index, subject in enumerate(unique_subjects):
        mask = np.asarray([str(value) == subject for value in subjects])
        if not mask.any():
            continue
        axes[1].scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=6,
            alpha=0.6,
            linewidths=0,
            color=FigureStyle.color(index),
            label=f"{subject} (n={int(mask.sum())})",
        )
    for ax, title in (
        (axes[0], "coloured by class"),
        (axes[1], "coloured by held-out participant"),
    ):
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(title)
        ax.legend(fontsize=7, markerscale=2.4)
        ax.grid(alpha=0.15)

    fig.suptitle(
        f"EXPLORATORY: t-SNE of held-out {embedding_source}s (identical projection, two "
        "colourings)",
        y=0.98,
    )
    caveat(
        fig,
        f"Exploratory only. perplexity={effective:g}, init=pca, seed={seed}, source="
        f"{checkpoint_id or 'unspecified'}. t-SNE preserves neighbourhoods, not distances or "
        "cluster sizes; apparent separation is a property of a non-linear projection and is "
        "not evidence of classification performance or of a clinical phenotype. Structure in "
        "the right panel that is absent from the left means the representation encodes "
        "participant identity.",
    )
    settings = {
        "method": "sklearn.manifold.TSNE",
        "n_components": 2,
        "perplexity": effective,
        "requested_perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": seed,
        "n_samples": n_samples,
        "embedding_dim": int(matrix.shape[1]),
        "embedding_source": embedding_source,
        "source_checkpoints": checkpoint_id,
        "status": "EXPLORATORY",
    }
    paths = save_figure(fig, output_dir, stem)
    write_csv(
        output_dir / f"{stem}_projection.csv",
        pd.DataFrame(
            {
                "tsne_1": projected[:, 0],
                "tsne_2": projected[:, 1],
                "true_class": [class_names[int(label)] for label in labels],
                "subject_id": [str(value) for value in subjects],
            }
        ),
    )
    return paths, settings
