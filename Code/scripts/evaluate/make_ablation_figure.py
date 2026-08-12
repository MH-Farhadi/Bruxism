"""Regenerate the modality-ablation manuscript figure for ``Paper/K_Farhadi_Paper_Bruxism``.

One figure depends on the modality ablation run and must be rebuilt whenever that run
changes::

    Figures/modality_ablation           audio-only / EMG-only / fusion, both tasks

Everything is derived from the run bundle's ``predictions.parquet``, so the figure cannot
disagree with the ledger it depicts. No number in it is typed by hand.

The four panels answer RQ2 in the order the question is asked. Panel (a) is the level
comparison, panel (b) is the paired fusion-minus-EMG difference that the level comparison
hides, panel (c) shows which classes the audio branch can rank on its own -- the "is the
benefit concentrated in chewing?" half of the question -- and panel (d) is the quiet-rest
false-alarm analogue, the one quantity on which the audio branch helps in every seed.

Usage, from ``Code/``::

    python scripts/evaluate/make_ablation_figure.py \
        --run-dir outputs/runs/modality_and_no_chewing_20260810T020642_cead62e4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bruxism.data.labels import get_task  # noqa: E402
from bruxism.evaluation.metrics import (  # noqa: E402
    PredictionLedger,
    one_vs_rest_auc,
    subject_level_summary,
)
from bruxism.visualization.paper_figures import FigureStyle, save_figure  # noqa: E402

#: Ledger identifiers in the order the manuscript names them, with manuscript wording.
#: Figures must never show code identifiers such as ``instructed_grinding``.
CLASS_LABELS: dict[str, str] = {
    "rest": "Rest",
    "movement": "Movement",
    "clench": "Clenching",
    "instructed_grinding": "Grinding",
    "chewing": "Chewing",
}
#: Ledger modality identifiers in the order they are plotted, with manuscript wording.
MODALITIES: tuple[str, ...] = ("audio_only", "emg_only", "fusion")
MODALITY_LABELS: dict[str, str] = {
    "audio_only": "Audio only",
    "emg_only": "EMG only",
    "fusion": "EMG + audio",
}
MODALITY_COLORS: dict[str, str] = {
    name: FigureStyle.color(index) for index, name in enumerate(MODALITIES)
}
#: Tasks in plotting order, with the short labels the axis uses.
TASKS: tuple[str, ...] = ("five_class", "no_chewing_four_class")
TASK_LABELS: dict[str, str] = {
    "five_class": "Five-class",
    "no_chewing_four_class": "Chewing removed\n(four-class)",
}

DEFAULT_OUTPUT = Path("../Paper/K_Farhadi_Paper_Bruxism/Figures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed ablation bundle.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--auc-task",
        default="five_class",
        help="Task whose per-class AUC panel is drawn.",
    )
    return parser


def load_ledger(run_dir: Path) -> tuple[pd.DataFrame, list[int]]:
    frame = pd.read_parquet(run_dir / "predictions.parquet")
    seeds = sorted(int(seed) for seed in frame["seed"].unique())
    return frame, seeds


def _subject_macro_f1(frame: pd.DataFrame, task_id: str) -> dict[str, float]:
    """Macro-F1 per held-out participant for one condition of one task."""
    names = tuple(get_task(task_id).class_names)
    summary = subject_level_summary(
        PredictionLedger(frame=frame.reset_index(drop=True), class_names=names)
    )
    return {subject: entry["macro_f1"] for subject, entry in summary["per_subject"].items()}


def collect(frame: pd.DataFrame, seeds: list[int]) -> dict[tuple[str, str, int], dict[str, float]]:
    """Participant macro-F1 keyed by (task, modality, seed)."""
    table: dict[tuple[str, str, int], dict[str, float]] = {}
    for task_id in TASKS:
        for modality in MODALITIES:
            for seed in seeds:
                subset = frame[
                    (frame["task_id"] == task_id)
                    & (frame["modality"] == modality)
                    & (frame["seed"] == seed)
                ]
                if subset.empty:
                    continue
                table[(task_id, modality, seed)] = _subject_macro_f1(subset, task_id)
    return table


def panel_levels(
    ax: plt.Axes, table: dict[tuple[str, str, int], dict[str, float]], seeds: list[int]
) -> None:
    """(a) Participant-level macro-F1 by modality, one group per task."""
    width = 0.26
    group = np.arange(len(TASKS), dtype=float)
    for index, modality in enumerate(MODALITIES):
        offset = (index - 1) * width
        heights, spreads = [], []
        for task_id in TASKS:
            per_seed = [
                float(np.mean(list(table[(task_id, modality, seed)].values()))) for seed in seeds
            ]
            heights.append(float(np.mean(per_seed)))
            spreads.append(per_seed)
        ax.bar(
            group + offset,
            heights,
            width=width,
            color=MODALITY_COLORS[modality],
            alpha=0.85,
            label=MODALITY_LABELS[modality],
        )
        for position, values, height in zip(group + offset, spreads, heights, strict=True):
            ax.plot(
                np.full(len(values), position),
                values,
                "o",
                markersize=3.0,
                markerfacecolor="white",
                markeredgecolor="#333333",
                markeredgewidth=0.7,
                zorder=3,
            )
            top = max(height, max(values))
            ax.text(position, top + 0.035, f"{height:.3f}", ha="center", fontsize=7)
    ax.set_xticks(group)
    ax.set_xticklabels([TASK_LABELS[task_id] for task_id in TASKS])
    ax.set_ylim(0, 1.16)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Macro F1 (participant-level mean)")
    ax.legend(loc="upper center", fontsize=7.5, ncol=3, columnspacing=1.0, handletextpad=0.4)
    ax.set_title("(a) Modality level", loc="left", fontsize=9)


def panel_paired(
    ax: plt.Axes, table: dict[tuple[str, str, int], dict[str, float]], seeds: list[int]
) -> list[str]:
    """(b) Paired fusion-minus-EMG-only difference, one point per participant and seed."""
    subjects = sorted(table[(TASKS[0], "fusion", seeds[0])])
    span = 0.62
    extremes: list[float] = []
    group_means: list[float] = []
    for task_index, task_id in enumerate(TASKS):
        means = []
        for subject_index, subject in enumerate(subjects):
            centre = task_index + (subject_index - (len(subjects) - 1) / 2) * (
                span / max(len(subjects) - 1, 1)
            )
            values = [
                table[(task_id, "fusion", seed)][subject]
                - table[(task_id, "emg_only", seed)][subject]
                for seed in seeds
            ]
            means.extend(values)
            extremes.extend(values)
            ax.plot(
                np.full(len(values), centre),
                values,
                "o",
                markersize=3.6,
                color=FigureStyle.color(subject_index),
                markeredgecolor="#333333",
                markeredgewidth=0.4,
                zorder=3,
                label=f"P{subject_index + 1}" if task_index == 0 else None,
            )
        mean = float(np.mean(means))
        group_means.append(mean)
        ax.hlines(
            mean,
            task_index - span / 2 - 0.09,
            task_index + span / 2 + 0.09,
            color="#333333",
            linewidth=1.4,
            zorder=4,
        )
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xticks(np.arange(len(TASKS), dtype=float))
    ax.set_xticklabels([TASK_LABELS[task_id] for task_id in TASKS])
    ax.set_xlim(-0.55, len(TASKS) - 0.45)
    low, high = min(extremes), max(extremes)
    pad = 0.09 * (high - low)
    ax.set_ylim(low - pad, high + 4.6 * pad)
    for task_index, mean in enumerate(group_means):
        ax.annotate(
            f"mean {mean:+.3f}",
            xy=(task_index, 0.83),
            xycoords=("data", "axes fraction"),
            ha="center",
            fontsize=7.5,
            color="#333333",
        )
    ax.set_ylabel(r"$\Delta$ macro F1 (EMG + audio $-$ EMG only)")
    ax.legend(
        loc="upper center",
        fontsize=7,
        ncol=5,
        columnspacing=0.7,
        handletextpad=0.15,
        borderpad=0.2,
    )
    ax.set_title("(b) Paired audio contribution", loc="left", fontsize=9)
    return subjects


def panel_class_auc(ax: plt.Axes, frame: pd.DataFrame, task_id: str, seeds: list[int]) -> None:
    """(c) Per-class one-vs-rest AUC by modality, averaged over seeds."""
    names = tuple(get_task(task_id).class_names)
    width = 0.26
    group = np.arange(len(names), dtype=float)
    for index, modality in enumerate(MODALITIES):
        heights = []
        for name in names:
            per_seed = []
            for seed in seeds:
                subset = frame[
                    (frame["task_id"] == task_id)
                    & (frame["modality"] == modality)
                    & (frame["seed"] == seed)
                ].reset_index(drop=True)
                ledger = PredictionLedger(frame=subset, class_names=names)
                value = one_vs_rest_auc(ledger.y_true, ledger.y_score, names)["roc_auc_per_class"][
                    name
                ]
                if value is not None:
                    per_seed.append(value)
            heights.append(float(np.mean(per_seed)))
        ax.bar(
            group + (index - 1) * width,
            heights,
            width=width,
            color=MODALITY_COLORS[modality],
            alpha=0.85,
        )
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
    ax.annotate(
        "chance",
        xy=(1.0, 0.5),
        xycoords=("axes fraction", "data"),
        xytext=(3, 0),
        textcoords="offset points",
        va="center",
        fontsize=7,
        color="#888888",
    )
    ax.set_xticks(group)
    ax.set_xticklabels([CLASS_LABELS[name] for name in names], rotation=30, ha="right")
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("One-vs-rest AUC")
    ax.set_title("(c) What each modality can rank (five-class)", loc="left", fontsize=9)


#: Classes that make up the collapsed tooth-contact group, so a quiet-rest window landing
#: in either of them is the paper's closest available analogue of a false alarm.
TOOTH_CONTACT: tuple[str, ...] = ("clench", "instructed_grinding")


def panel_false_alarm(ax: plt.Axes, frame: pd.DataFrame, seeds: list[int]) -> None:
    """(d) Share of quiet-rest windows predicted as clenching or grinding."""
    width = 0.26
    group = np.arange(len(TASKS), dtype=float)
    for index, modality in enumerate(MODALITIES):
        offset = (index - 1) * width
        heights, spreads = [], []
        for task_id in TASKS:
            per_seed = []
            for seed in seeds:
                rest = frame[
                    (frame["task_id"] == task_id)
                    & (frame["modality"] == modality)
                    & (frame["seed"] == seed)
                    & (frame["true_class"] == "rest")
                ]
                per_seed.append(float(rest["predicted_class"].isin(TOOTH_CONTACT).mean()))
            heights.append(float(np.mean(per_seed)))
            spreads.append(per_seed)
        ax.bar(
            group + offset,
            heights,
            width=width,
            color=MODALITY_COLORS[modality],
            alpha=0.85,
        )
        for position, values, height in zip(group + offset, spreads, heights, strict=True):
            ax.plot(
                np.full(len(values), position),
                values,
                "o",
                markersize=3.0,
                markerfacecolor="white",
                markeredgecolor="#333333",
                markeredgewidth=0.7,
                zorder=3,
            )
            top = max(height, max(values))
            ax.text(position, top + 0.009, f"{height:.1%}", ha="center", fontsize=7)
    ax.set_xticks(group)
    ax.set_xticklabels([TASK_LABELS[task_id] for task_id in TASKS])
    ax.set_ylabel("Quiet-rest windows read as\nclenching or grinding")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_ylim(0, 0.245)
    ax.set_title("(d) Quiet-rest false-alarm analogue", loc="left", fontsize=9)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    frame, seeds = load_ledger(run_dir)
    table = collect(frame, seeds)

    FigureStyle.apply()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    panel_levels(axes[0][0], table, seeds)
    subjects = panel_paired(axes[0][1], table, seeds)
    panel_class_auc(axes[1][0], frame, args.auc_task, seeds)
    panel_false_alarm(axes[1][1], frame, seeds)
    fig.tight_layout()
    save_figure(fig, out, "modality_ablation")

    bundle = json.loads((run_dir / "run_bundle.json").read_text())
    provenance = {
        "run_id": bundle["run_id"],
        "config_hash": bundle["config_hash"],
        "manifest_hash": bundle["manifest_hash"],
        "window_index_hash": bundle["window_index_hash"],
        "seeds": seeds,
        "subjects": subjects,
        "tasks": list(TASKS),
        "modalities": list(MODALITIES),
        "n_predictions": int(len(frame)),
        "figures": ["modality_ablation"],
        "note": (
            "Every condition shares the windows, folds, seeds and fixed training "
            "configuration of this run; only the modality differs, and the unused branch "
            "is not constructed. Panel (a) bars are the mean over five participants and "
            "three seeds, markers the three seed means. Panel (b) is the paired "
            "fusion-minus-EMG-only difference for each participant and seed, so the "
            "vertical spread is the quantity the panel (a) bars hide."
        ),
    }
    (out / "modality_ablation_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote modality_ablation (PNG + PDF) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
