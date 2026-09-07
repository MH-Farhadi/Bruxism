"""Fig. 3 of main_v9: (a) counts, (b) row-normalised recall, (c) macro-F1 per held-out
participant -- the v8 diagnostics figure redrawn as one labelled multipart figure at a
size that stays legible at page width. Same ledger, same seed rule and same computations
as Code/scripts/evaluate/make_manuscript_figures.py; only the layout differs."""
import sys
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path("/home/kye/Desktop/Depo/Code/Bruxism/Code")
spec = importlib.util.spec_from_file_location("mmf", CODE / "scripts/evaluate/make_manuscript_figures.py")
mmf = importlib.util.module_from_spec(spec); spec.loader.exec_module(mmf)
from bruxism.evaluation.metrics import PredictionLedger, confusion_matrices, subject_level_summary
from bruxism.visualization.paper_figures import FigureStyle

RUN = CODE / "outputs/runs/modality_and_no_chewing_20260810T020642_cead62e4"
OUT = Path("/home/kye/Desktop/Depo/Code/Bruxism/Paper/K_Farhadi_Paper_Bruxism/Figures/v9")
frame, seeds = mmf.load_ledger(RUN, "five_class")
frame = frame[frame["modality"] == "emg_only"].reset_index(drop=True)
assert len(frame) == 18519, len(frame)
single = frame[frame["seed"] == seeds[0]].reset_index(drop=True)
CLASS_IDS, labels = mmf.CLASS_IDS, mmf._labels(mmf.CLASS_IDS)

FigureStyle.apply()
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
fig = plt.figure(figsize=(7.16, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.35], wspace=0.36, left=0.075, right=0.985, top=0.90, bottom=0.30)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

# (a), (b): one model per fold at the lowest seed index (fixed rule, see provenance note)
ledger = PredictionLedger(frame=single, class_names=CLASS_IDS)
m = confusion_matrices(ledger.y_true, ledger.y_pred, len(CLASS_IDS))
raw, norm = m["raw"], m["row_normalized"]
for pos, (ax, matrix, title, fmt) in enumerate(((axes[0], raw, "(a) Counts", "d"), (axes[1], norm, "(b) Recall", ".2f"))):
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(labels, rotation=35, ha="right"); ax.set_yticklabels(labels if pos == 0 else [])
    ax.set_xlabel("Predicted")
    if pos == 0: ax.set_ylabel("True")
    ax.set_title(title, loc="left"); ax.grid(False)
    for s in ("top", "right"): ax.spines[s].set_visible(True)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, format(matrix[i, j], fmt) if fmt == ".2f" else f"{matrix[i, j]:,}",
                    ha="center", va="center", fontsize=6.2, color="white" if norm[i, j] > 0.55 else "black")

# (c): bars = mean over the three seeds, markers = individual seeds
per_seed, counts = {}, {}
for seed in seeds:
    subset = frame[frame["seed"] == seed].reset_index(drop=True)
    summary = subject_level_summary(PredictionLedger(frame=subset, class_names=CLASS_IDS))
    per_seed[seed] = {s: e["macro_f1"] for s, e in summary["per_subject"].items()}
    counts = {s: e["n_samples"] for s, e in summary["per_subject"].items()}
subjects = sorted(counts)
means = [float(np.mean([per_seed[s][sub] for s in seeds])) for sub in subjects]
ax = axes[2]; pos_ = np.arange(len(subjects), dtype=float)
ax.bar(pos_, means, color=[FigureStyle.color(i) for i in range(len(subjects))], width=0.62, alpha=0.85)
for k, sub in enumerate(subjects):
    vals = [per_seed[s][sub] for s in seeds]
    ax.plot(np.full(len(vals), pos_[k]), vals, "o", markersize=2.8, markerfacecolor="white",
            markeredgecolor="#333333", markeredgewidth=0.7, zorder=3)
    ax.text(pos_[k], max(vals) + 0.03, f"{means[k]:.3f}", ha="center", fontsize=6.5)
grand = float(np.mean(means))
ax.axhline(grand, color="#333333", linestyle="--", linewidth=0.9)
ax.text(-0.45, 0.965, f"dashed line: mean {grand:.3f}", ha="left", va="top", fontsize=6.3, color="#333333")
ax.set_xticks(pos_); ax.set_xticklabels([f"P{i + 1}\nn={counts[sub]:,}" for i, sub in enumerate(subjects)], fontsize=6.5)
ax.set_ylim(0, 1.0); ax.set_xlim(-0.55, len(subjects) - 0.45); ax.set_xlabel("Held-out participant (held-out windows)"); ax.set_ylabel("Macro-F1")
ax.set_title("(c) Per held-out participant", loc="left")
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig3_errors.{ext}", dpi=600 if ext == "png" else None)
print("raw counts:\n", raw, "\nmeans:", [round(v, 3) for v in means], "grand", round(grand, 3), "counts", counts)
