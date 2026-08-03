"""Manuscript figures and tables, generated exclusively from saved artifacts.

Every function here takes a saved prediction ledger, metrics JSON, fold-outcome JSON or
benchmark JSON. None of them re-runs a model, re-reads raw data or accepts a hand-entered
number, so a figure can never disagree with the ledger it claims to depict.

Regenerating the whole bundle is one command::

    bruxism-report --runs-root outputs/runs --output-root outputs/paper_bundle

Figures are written as PNG (300 dpi) and PDF. The matplotlib backend is forced to ``Agg``
and ``plt.show()`` is never called, so the generator runs headless.
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

from bruxism.evaluation.metrics import PredictionLedger, confusion_matrices  # noqa: E402
from bruxism.utils.io import write_csv, write_json  # noqa: E402
from bruxism.utils.logging import get_logger  # noqa: E402

__all__ = [
    "FigureStyle",
    "caveat",
    "plot_confusion_matrices",
    "plot_flow_diagram",
    "plot_modality_comparison",
    "plot_per_participant",
    "plot_pr_curves",
    "plot_roc_curves",
    "plot_training_curves",
    "plot_tsne",
    "save_figure",
    "write_latex_tables",
]

logger = get_logger(__name__)


class FigureStyle:
    """Shared, colour-blind-safe styling for every figure in the bundle."""

    #: Okabe-Ito palette: distinguishable under the common colour-vision deficiencies and
    #: in greyscale print.
    PALETTE: tuple[str, ...] = (
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#CC79A7",
        "#D55E00",
        "#56B4E9",
        "#F0E442",
        "#000000",
    )
    DPI = 300
    FIGSIZE = (7.0, 5.0)

    @classmethod
    def apply(cls) -> None:
        plt.rcParams.update(
            {
                "figure.dpi": 120,
                "savefig.dpi": cls.DPI,
                "savefig.bbox": "tight",
                "font.size": 9,
                "axes.titlesize": 10,
                "axes.labelsize": 9,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "grid.linewidth": 0.5,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "legend.frameon": False,
                "figure.autolayout": False,
            }
        )

    @classmethod
    def color(cls, index: int) -> str:
        return cls.PALETTE[index % len(cls.PALETTE)]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    """Write a figure as PNG and PDF, then close it. Never calls ``plt.show()``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in (".png", ".pdf"):
        path = output_dir / f"{stem}{suffix}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    logger.debug("wrote figure %s", stem)
    return written


def caveat(fig: plt.Figure, text: str) -> None:
    """Stamp a scope caveat below a figure so it survives being copied into a slide deck.

    The text is positioned below the lowest drawn element (rotated tick labels included)
    rather than at a fixed offset, so it never collides with long class names.
    """
    fig.canvas.draw()
    renderer = getattr(fig.canvas, "get_renderer", lambda: None)()
    lowest = 1.0
    for ax in fig.axes:
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            continue
        lowest = min(lowest, bbox.transformed(fig.transFigure.inverted()).y0)
    fig.text(
        0.5,
        lowest - 0.035,
        text,
        ha="center",
        va="top",
        fontsize=6.5,
        style="italic",
        color="#555555",
        wrap=True,
    )


# ------------------------------------------------------------------ confusion ---


def plot_confusion_matrices(
    ledger: PredictionLedger, output_dir: Path, *, stem: str = "confusion_matrix"
) -> list[Path]:
    """Raw-count and row-normalised confusion matrices, side by side."""
    FigureStyle.apply()
    names = list(ledger.class_names)
    matrices = confusion_matrices(ledger.y_true, ledger.y_pred, len(names))
    raw, normalized = matrices["raw"], matrices["row_normalized"]

    fig, axes = plt.subplots(1, 2, figsize=(3.6 * len(names) / 2 + 4.0, 5.2))
    for position, (ax, matrix, title, fmt) in enumerate(
        (
            (axes[0], raw, "Counts", "d"),
            (axes[1], normalized, "Row-normalised (recall)", ".2f"),
        )
    ):
        image = ax.imshow(
            normalized, cmap="Blues", vmin=0, vmax=1, interpolation="nearest", aspect="equal"
        )
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right")
        ax.set_yticklabels(names)
        ax.set_xlabel("Predicted")
        if position == 0:
            ax.set_ylabel("True")
        else:
            ax.set_yticklabels([])
        ax.set_title(title)
        ax.grid(False)
        for i in range(len(names)):
            for j in range(len(names)):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if normalized[i, j] > 0.55 else "black",
                )
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Row-normalised rate")
    fig.suptitle("Held-out confusion matrices (pooled across leave-one-subject-out folds)", y=0.99)
    caveat(
        fig,
        "Pooled windows are descriptive only: windows within a participant and recording "
        "are correlated. Participant-level results are the primary evidence.",
    )
    return save_figure(fig, output_dir, stem)


# ---------------------------------------------------------------------- curves ---


def plot_roc_curves(
    curves: dict[str, Any], output_dir: Path, *, stem: str = "roc_curves", title: str = ""
) -> list[Path]:
    """One-vs-rest ROC curves with per-class AUC in the legend."""
    FigureStyle.apply()
    fig, ax = plt.subplots(figsize=FigureStyle.FIGSIZE)
    plotted = 0
    for index, (name, entry) in enumerate(sorted(curves.items())):
        if not entry.get("available"):
            continue
        ax.plot(
            entry["fpr"],
            entry["tpr"],
            color=FigureStyle.color(index),
            linewidth=1.6,
            label=f"{name} (AUC {entry['roc_auc']:.3f}, n+={entry['positive_support']})",
        )
        plotted += 1
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1.0, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_title(title or "One-vs-rest ROC, held-out participants")
    ax.legend(loc="lower right", fontsize=7)
    if plotted == 0:
        ax.text(0.5, 0.5, "no class had both positive and negative examples", ha="center")
    caveat(fig, "Curves are pooled over held-out windows and are descriptive.")
    return save_figure(fig, output_dir, stem)


def plot_pr_curves(
    curves: dict[str, Any], output_dir: Path, *, stem: str = "pr_curves", title: str = ""
) -> list[Path]:
    """One-vs-rest precision-recall curves with average precision in the legend."""
    FigureStyle.apply()
    fig, ax = plt.subplots(figsize=FigureStyle.FIGSIZE)
    for index, (name, entry) in enumerate(sorted(curves.items())):
        if not entry.get("available"):
            continue
        ax.plot(
            entry["recall"],
            entry["precision"],
            color=FigureStyle.color(index),
            linewidth=1.6,
            label=f"{name} (AP {entry['average_precision']:.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_title(title or "One-vs-rest precision-recall, held-out participants")
    ax.legend(loc="lower left", fontsize=7)
    caveat(
        fig,
        "Precision-recall is the more informative view under this class imbalance; the "
        "no-skill baseline differs per class and equals that class's prevalence.",
    )
    return save_figure(fig, output_dir, stem)


# ------------------------------------------------------------- per participant ---


def plot_per_participant(
    metrics: dict[str, Any],
    output_dir: Path,
    *,
    metric: str = "macro_f1",
    stem: str = "per_participant",
) -> list[Path]:
    """Primary figure: one bar per participant, with the sample mean marked.

    Participants are the unit of generalisation, so this is the figure the manuscript
    should lead with.
    """
    FigureStyle.apply()
    per_subject = metrics.get("subject_level", {}).get("per_subject", {})
    if not per_subject:
        raise ValueError("metrics contain no per-participant results")

    subjects = sorted(per_subject)
    values = [per_subject[subject].get(metric) for subject in subjects]
    counts = [per_subject[subject].get("n_samples", 0) for subject in subjects]
    usable = [value for value in values if value is not None]

    fig, ax = plt.subplots(figsize=(max(5.0, 1.1 * len(subjects) + 3.0), 4.4))
    positions = np.arange(len(subjects))
    ax.bar(
        positions,
        [value if value is not None else 0.0 for value in values],
        color=[FigureStyle.color(i) for i in range(len(subjects))],
        width=0.62,
    )
    if usable:
        # The mean is annotated on the line itself rather than in a legend, so it cannot
        # collide with a bar label wherever the bars happen to sit.
        mean = float(np.mean(usable))
        ax.axhline(mean, color="#333333", linestyle="--", linewidth=1.1)
        ax.annotate(
            f"mean {mean:.3f}",
            xy=(1.0, mean),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )
    for raw_position, value, count in zip(positions, values, counts):
        position = float(raw_position)
        if value is None:
            ax.text(position, 0.02, "n/a", ha="center", fontsize=8)
            continue
        ax.text(position, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
        ax.text(position, 0.02, f"n={count}", ha="center", fontsize=7, color="white")
    ax.set_xticks(positions)
    ax.set_xticklabels(subjects)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_xlabel("Held-out participant")
    ax.set_title(f"Per-participant {metric.replace('_', ' ')} (leave-one-subject-out)")
    caveat(
        fig,
        "Each bar is the participant's own held-out fold. Five participants cannot "
        "characterise population variability; no generalisation claim follows from the mean.",
    )
    return save_figure(fig, output_dir, stem)


# ------------------------------------------------------------ training curves ---


def plot_training_curves(
    fold_outcomes: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    stem: str = "training_curves",
) -> list[Path]:
    """Loss and macro-F1 per epoch, from the inner-fold histories.

    The outer held-out participant appears in **no** curve: inner-fold curves are measured
    on inner-validation participants, and the refit curves are labelled as training-fit
    diagnostics.
    """
    FigureStyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    drawn = 0
    for index, outcome in enumerate(fold_outcomes):
        history = outcome.get("training_history") or []
        if not history:
            continue
        epochs = [record["epoch"] for record in history]
        label = f"fold {outcome.get('outer_fold')} (held out {outcome.get('test_subject')})"
        colour = FigureStyle.color(index)
        axes[0].plot(
            epochs, [r["train_loss"] for r in history], color=colour, linewidth=1.4, label=label
        )
        axes[0].plot(
            epochs, [r["val_loss"] for r in history], color=colour, linewidth=1.0, linestyle="--"
        )
        axes[1].plot(
            epochs, [r["val_macro_f1"] for r in history], color=colour, linewidth=1.4, label=label
        )
        drawn += 1

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss (solid: training, dashed: fit diagnostic)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Macro F1")
    if drawn:
        axes[0].legend(fontsize=7)
    scope = {outcome.get("history_scope", "unknown") for outcome in fold_outcomes}
    caveat(
        fig,
        "The outer held-out participant contributes to no curve here. Refit histories "
        f"(scope: {', '.join(sorted(scope))}) are measured on training data and are fit "
        "diagnostics, not validation.",
    )
    return save_figure(fig, output_dir, stem)


# ----------------------------------------------------------------- comparison ---


def plot_modality_comparison(
    table: pd.DataFrame,
    output_dir: Path,
    *,
    metric: str = "macro_f1_mean",
    stem: str = "modality_comparison",
) -> list[Path]:
    """Grouped bars: modality on the x-axis, one group per task."""
    FigureStyle.apply()
    if table.empty or metric not in table.columns:
        raise ValueError(f"comparison table lacks the column {metric!r}")

    pivot = table.pivot_table(index="task_id", columns="modality", values=metric, aggfunc="mean")
    tasks = list(pivot.index)
    modalities = list(pivot.columns)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.7 * len(tasks) + 3.0), 4.6))
    width = 0.8 / max(len(modalities), 1)
    for index, modality in enumerate(modalities):
        offsets = np.arange(len(tasks)) + index * width - 0.4 + width / 2
        values = pivot[modality].to_numpy(dtype=float)
        ax.bar(
            offsets,
            np.nan_to_num(values),
            width=width,
            label=modality,
            color=FigureStyle.color(index),
        )
        for offset, value in zip(offsets, values):
            if not np.isnan(value):
                ax.text(offset, value + 0.012, f"{value:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=8)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(0, 1.12)
    ax.set_title("Modality comparison (participant-level mean)")
    ax.legend(title="modality", fontsize=8)
    caveat(
        fig,
        "All conditions share identical windows, folds, seeds and selection budget; only "
        "the modality differs. Compare the five-class and no-chewing groups to see whether "
        "the audio benefit survives removal of eating.",
    )
    return save_figure(fig, output_dir, stem)


# ---------------------------------------------------------------------- t-SNE ---


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path,
    *,
    seed: int = 0,
    perplexity: float = 30.0,
    checkpoint_id: str = "",
    stem: str = "tsne",
) -> tuple[list[Path], dict[str, Any]]:
    """Exploratory 2-D t-SNE of held-out embeddings.

    Returns the figure paths and the exact settings used, which are saved next to the
    figure so the plot is reproducible. Cluster appearance in a t-SNE plot is **not**
    independent validation and the figure says so on its face.
    """
    from sklearn.manifold import TSNE

    FigureStyle.apply()
    n_samples = int(embeddings.shape[0])
    effective_perplexity = float(min(perplexity, max(5.0, (n_samples - 1) / 3.0)))
    settings = {
        "method": "sklearn.manifold.TSNE",
        "n_components": 2,
        "perplexity": effective_perplexity,
        "requested_perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": seed,
        "n_samples": n_samples,
        "embedding_dim": int(embeddings.shape[1]),
        "source_checkpoint": checkpoint_id,
        "status": "EXPLORATORY",
        "note": (
            "Computed on held-out embeddings only. Visual cluster separation is a "
            "descriptive property of a non-linear projection and is not evidence of "
            "classification performance."
        ),
    }
    projected = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(np.asarray(embeddings, dtype=np.float64))

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for index, name in enumerate(class_names):
        mask = labels == index
        if not mask.any():
            continue
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=7,
            alpha=0.65,
            color=FigureStyle.color(index),
            label=f"{name} (n={int(mask.sum())})",
            linewidths=0,
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("EXPLORATORY: held-out embeddings, t-SNE projection")
    ax.legend(fontsize=7, markerscale=2)
    ax.grid(alpha=0.15)
    caveat(
        fig,
        f"Exploratory only. perplexity={effective_perplexity:g}, init=pca, seed={seed}, "
        f"checkpoint={checkpoint_id or 'unspecified'}. Cluster appearance is not validation.",
    )
    paths = save_figure(fig, output_dir, stem)
    write_json(output_dir / f"{stem}_settings.json", settings)
    return paths, settings


# ------------------------------------------------------------------ flow / counts ---


def plot_flow_diagram(
    counts: pd.DataFrame, output_dir: Path, *, stem: str = "sample_flow"
) -> list[Path]:
    """Stacked bars of windows per participant and class -- the sample-flow figure."""
    FigureStyle.apply()
    if counts.empty:
        raise ValueError("no sample counts to plot")
    pivot = counts.pivot_table(
        index="subject_id", columns="task_family", values="n_windows", aggfunc="sum", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(pivot) + 3.5), 4.6))
    bottom = np.zeros(len(pivot))
    for index, column in enumerate(pivot.columns):
        values = pivot[column].to_numpy(dtype=float)
        ax.bar(
            pivot.index,
            values,
            bottom=bottom,
            label=str(column),
            color=FigureStyle.color(index),
            width=0.62,
        )
        bottom += values
    for position, total in enumerate(bottom):
        ax.text(
            float(position),
            float(total + max(bottom) * 0.015),
            f"{int(total)}",
            ha="center",
            fontsize=8,
        )
    ax.set_ylabel("Windows after exclusions")
    ax.set_xlabel("Participant")
    ax.set_title("Analysable windows per participant and task family")
    ax.legend(fontsize=8, ncols=2)
    caveat(
        fig,
        "Counts follow the approved segmentation policy: windows lie wholly inside a "
        "trigger-active interval, clear of the transition and startup guards.",
    )
    return save_figure(fig, output_dir, stem)


# ------------------------------------------------------------------ LaTeX tables ---


def _escape(value: Any) -> str:
    text = "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(old, new)
    return text


def dataframe_to_latex(
    frame: pd.DataFrame, *, caption: str, label: str, float_format: str = "{:.3f}"
) -> str:
    """Render a DataFrame as a booktabs LaTeX table.

    Generated from the frame, never hand-edited: a number in the manuscript that does not
    appear in the corresponding CSV is a bug.
    """
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda v: "n/a" if pd.isna(v) else float_format.format(v)
            )
    header = " & ".join(_escape(c) for c in formatted.columns) + r" \\"
    body = "\n".join(
        " & ".join(_escape(v) for v in row) + r" \\" for row in formatted.astype(str).to_numpy()
    )
    alignment = "l" * len(formatted.columns)
    return "\n".join(
        [
            r"% Generated by bruxism.visualization.paper_figures -- do not edit by hand.",
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
            header,
            r"\midrule",
            body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def write_latex_tables(tables: dict[str, tuple[pd.DataFrame, str]], output_dir: Path) -> list[Path]:
    """Write each ``name -> (frame, caption)`` as both CSV and a ``.tex`` table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, (frame, caption) in sorted(tables.items()):
        written.append(write_csv(output_dir / f"{name}.csv", frame))
        tex = output_dir / f"{name}.tex"
        tex.write_text(
            dataframe_to_latex(frame, caption=caption, label=f"tab:{name}"), encoding="utf-8"
        )
        written.append(tex)
    return written


def write_latex_macros(values: dict[str, Any], output_dir: Path, *, stem: str = "macros") -> Path:
    """Emit ``\\newcommand`` macros so the manuscript can cite generated numbers by name.

    Using ``\\bruxAccuracy`` in the LaTeX source instead of a literal number makes it
    impossible for the manuscript to drift from the artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by bruxism.visualization.paper_figures -- do not edit by hand.",
        "% Every value below is derived from a saved prediction ledger.",
    ]
    for key in sorted(values):
        raw = values[key]
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            rendered = r"\textbf{[BLOCKED]}"
        elif isinstance(raw, float):
            rendered = f"{raw:.3f}"
        else:
            rendered = _escape(raw)
        macro = "brux" + "".join(part.capitalize() for part in key.split("_"))
        lines.append(rf"\newcommand{{\{macro}}}{{{rendered}}}")
    path = output_dir / f"{stem}.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
