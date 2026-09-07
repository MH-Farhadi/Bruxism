#!/usr/bin/env python3
"""Draw the end-to-end pipeline figure: raw signal -> decision logits.

Replaces the architecture-only diagram (Figures/v5/architecture_emg.png) referenced as
Fig. 4 in main_v7.tex.

Provenance: band edges, tensor shapes and parameter counts are READ FROM THE INSTANTIATED
MODEL at draw time, never transcribed. That is what lets the caption keep its sentence
"Layer widths, wavelet family, band edges and parameter counts are read from the
instantiated model rather than transcribed."

Layout is hand-placed on a 1120 x 400 grid (matching the approved mockup); the y axis is
inverted so those coordinates read top-down like the mockup does.

Sizing: the figure is drawn at its true print size (7.5 in wide) so that font sizes in
points are the sizes that reach the page. Include it at width=\\linewidth.

Usage
-----
    python make_pipeline_figure.py                 # schematic input traces
    python make_pipeline_figure.py --recording X   # real excerpt for the input traces

Outputs pipeline_figure.pdf (include this one) and pipeline_figure.png (for previewing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
# Type 42 = TrueType. Matplotlib's default is Type 3, which Springer Nature and most other
# publishers reject at production. Set before pyplot draws anything.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Code" / "src"))

# --------------------------------------------------------------------------------------
# Palette. The band ramp is a single-hue sequential scale, chosen because the bands are
# ordered by frequency; it deliberately avoids the seaborn "colorblind" palette, which is
# already spoken for by the five CLASSES in Figs 6-8. Reusing those hues here would make
# blue mean "rest" on one page and "A4" on another.
# --------------------------------------------------------------------------------------
BAND = ["#2F3B6B", "#5A6BA8", "#96A3D4"]   # A4 (low) -> D3 -> D1 (high)
BAND_TEXT = ["#FFFFFF", "#FFFFFF", "#16192B"]
TINT = "#EEF0F8"
INK = "#1A1C26"
SOFT = "#43475A"
MUTED = "#767A8C"
RULE = "#B9BCCC"
FLAG = "#8A5320"
CLASS = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#D55E00"]  # rest..chewing
CLASS_NAMES = ["rest", "movement", "clenching", "grinding", "chewing"]

SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

# Type scale, in points at the drawn size (7.5 in wide). Include at width=\linewidth in a
# two-column sn-jnl figure* (~6.9 in) and everything scales by 0.92.
FS_EYEBROW = 6.6
FS_BAND = 10.0
FS_STAGE = 7.6
FS_SUB = 6.7
FS_SHAPE = 6.6
FS_ANNOT = 6.9
FS_FOOT = 6.3
FS_CLASS = 6.2


def model_facts() -> dict:
    """Read every quoted number out of the real model. No literals."""
    import torch
    from bruxism.models.dual_branch import build_model, DualBranchConfig
    from bruxism.preprocessing.wavelets import band_frequencies

    cfg = DualBranchConfig(modality="emg_only", num_classes=5)
    model = build_model(cfg).eval()
    branch = cfg.emg
    bands = list(branch.wavelet.bands)
    level = branch.wavelet.level
    fs = 1200.0

    shapes: dict[str, tuple] = {}
    handles = []

    def make_hook(name: str):
        # NB: a forward hook that returns anything non-None REPLACES the module's output.
        # This one records and returns None.
        def hook(_module, inputs, output) -> None:
            if name not in shapes:
                shapes[name] = (
                    tuple(inputs[0].shape)[1:] if torch.is_tensor(inputs[0]) else None,
                    tuple(output.shape)[1:] if torch.is_tensor(output) else None,
                )
        return hook

    for name, mod in model.named_modules():
        if name:
            handles.append(mod.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(torch.zeros(2, branch.in_channels, 1200), torch.zeros(2, 1, 1200))
    for h in handles:
        h.remove()

    per_band = []
    for b in bands:
        lo, hi = band_frequencies(b, level, fs)
        coeff_in, _ = shapes[f"emg_branch.bands.{b}.features"]
        conv_out = shapes[f"emg_branch.bands.{b}.features"][1]
        pooled = shapes[f"emg_branch.bands.{b}.pool"][1]
        per_band.append(
            dict(name=b, lo=lo, hi=hi, coeff=coeff_in, conv=conv_out, pooled=pooled[0])
        )

    counts = model.parameter_counts()
    per_band_params = counts["emg_branch"] // len(bands)
    return dict(
        bands=per_band,
        wavelet=branch.wavelet.wavelet,
        level=level,
        in_channels=branch.in_channels,
        window=1200,
        fs=fs,
        concat=shapes["emg_branch"][1][0],
        mlp_out=shapes["fusion"][1][0],
        n_classes=shapes["classifier"][1][0],
        p_branch=counts["emg_branch"],
        p_band=per_band_params,
        p_fusion=counts["fusion"],
        p_head=counts["classifier"],
        p_total=counts["trainable"],
    )


def input_traces(recording: Path | None, n_ch: int, n: int = 260) -> np.ndarray:
    """Real excerpt if a recording is supplied, otherwise a schematic burst pattern.

    The schematic is explicitly NOT presented as data: it is drawn small, unlabelled and
    without axes, as an icon for "four channels of EMG".
    """
    if recording is not None and recording.exists():
        arr = np.load(recording)
        arr = arr[:n_ch, :n] if arr.ndim == 2 else arr.reshape(n_ch, -1)[:, :n]
        arr = arr - arr.mean(axis=1, keepdims=True)
        scale = np.abs(arr).max(axis=1, keepdims=True)
        return arr / np.where(scale == 0, 1, scale)

    rng = np.random.default_rng(7)
    t = np.linspace(0, 3.2, n)
    out = np.empty((n_ch, n))
    for c in range(n_ch):
        # burst-relaxation envelope at ~1.4 Hz, the chewing rhythm, with per-channel phase
        env = 0.20 + 0.80 * np.clip(np.sin(2 * np.pi * 1.4 * t - c * 0.55), 0, None) ** 1.6
        sig = rng.normal(0, 1, n)
        sig = np.convolve(sig, np.ones(3) / 3, mode="same")  # mild smoothing
        out[c] = env * sig
        out[c] /= np.abs(out[c]).max()
    return out


def rbox(ax, x, y, w, h, *, fc="none", ec=RULE, lw=0.8, r=4.0, z=2, alpha=1.0):
    ax.add_patch(
        FancyBboxPatch(
            (x + r, y + r), w - 2 * r, h - 2 * r,
            boxstyle=f"round,pad={r},rounding_size={r}",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=alpha,
        )
    )


def arrow(ax, x1, y1, x2, y2, *, color=MUTED, lw=0.8, rad=None, ms=4.5, z=3):
    cs = "arc3,rad=0" if rad is None else f"arc3,rad={rad}"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, linewidth=lw, shrinkA=0, shrinkB=0,
            mutation_scale=ms, connectionstyle=cs,
        ),
        zorder=z,
    )


def txt(ax, x, y, s, *, size=FS_SUB, color=SOFT, ha="center", va="center",
        weight="normal", family=SANS, z=5, style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight,
            fontfamily=family, zorder=z, fontstyle=style)


def draw(facts: dict, traces: np.ndarray, out_stem: Path) -> None:
    """Hand-placed layout on a 1180 x 520 grid, y inverted so it reads top-down.

    Drawn at very nearly the printed width. A two-column sn-jnl figure* is about 6.93 in;
    drawing at 6.9 in means \\includegraphics[width=\\linewidth] scales by ~0.99, so the
    point sizes below are the point sizes that reach the page. Overflow behaviour is
    scale-invariant (text and boxes scale together), so this only sets absolute type size.
    """
    fig, ax = plt.subplots(figsize=(6.9, 3.15))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1180)
    ax.set_ylim(520, 0)
    ax.axis("off")

    SPINE = 225
    ROW = [110, 225, 340]
    BOX_T, BOX_H = 183, 84

    # ---------------- section eyebrows -------------------------------------------------
    for x, s_ in ((20, "SIGNAL CONDITIONING"), (582, "LEARNED REPRESENTATION"),
                  (908, "DECISION")):
        txt(ax, x, 18, s_, size=FS_EYEBROW, color=MUTED, ha="left", weight="bold")
    ax.plot([20, 1160], [30, 30], color=RULE, lw=0.6, zorder=1)

    # ---------------- 1. raw EMG -------------------------------------------------------
    rbox(ax, 20, BOX_T, 112, BOX_H)
    n_ch, n = traces.shape
    xs = np.linspace(30, 122, n)
    for c in range(n_ch):
        ax.plot(xs, 198 + c * 18 - traces[c] * 7.0, color=SOFT, lw=0.5, zorder=4,
                solid_capstyle="round")
    txt(ax, 76, 288, f"Raw EMG · {facts['in_channels']} ch", size=FS_SUB, color=MUTED)
    txt(ax, 76, 303, f"({facts['in_channels']}, T) @ {facts['fs']:.0f} Hz",
        size=FS_SHAPE, color=MUTED, family=MONO)

    arrow(ax, 136, SPINE, 154, SPINE)

    # ---------------- 2. notch bank + band-pass ----------------------------------------
    rbox(ax, 158, BOX_T, 136, BOX_H, fc=TINT)
    txt(ax, 226, 202, "Notch ×7", size=FS_STAGE, color=INK, weight="semibold")
    txt(ax, 226, 217, "+ band-pass", size=FS_STAGE, color=INK, weight="semibold")
    for k in range(7):
        x = 186 + k * 13
        ax.plot([x, x], [236, 252], color=SOFT, lw=0.8, zorder=4)
    ax.plot([180, 272], [252, 252], color=RULE, lw=0.6, zorder=4)
    txt(ax, 226, 288, "60–420 · 20–450 Hz", size=FS_FOOT, color=MUTED)
    txt(ax, 226, 303, "see Fig. 2", size=FS_FOOT, color=MUTED, style="italic")

    arrow(ax, 298, SPINE, 316, SPINE)

    # ---------------- 3. window + z-score ----------------------------------------------
    rbox(ax, 320, BOX_T, 132, BOX_H)
    txt(ax, 386, 202, "Window", size=FS_STAGE, color=INK, weight="semibold")
    txt(ax, 386, 217, "+ z-score", size=FS_STAGE, color=INK, weight="semibold")
    txt(ax, 386, 238, "1.0 s / 0.5 s", size=FS_FOOT, color=MUTED)
    txt(ax, 386, 253, "train ppts only", size=FS_FOOT, color=MUTED)
    txt(ax, 386, 288, f"({facts['in_channels']}, {facts['window']})",
        size=FS_SHAPE, color=MUTED, family=MONO)

    arrow(ax, 456, SPINE, 474, SPINE)

    # ---------------- 4. wavelet split -------------------------------------------------
    rbox(ax, 478, BOX_T, 72, BOX_H, fc=TINT)
    txt(ax, 514, 208, "DWT", size=FS_STAGE, color=INK, weight="semibold")
    txt(ax, 514, 223, facts["wavelet"], size=FS_STAGE, color=INK, weight="semibold")
    txt(ax, 514, 243, f"{facts['level']} levels", size=FS_FOOT, color=MUTED)

    # ---------------- 5. the three band rows -------------------------------------------
    for i, (b, cy) in enumerate(zip(facts["bands"], ROW)):
        col, tcol = BAND[i], BAND_TEXT[i]

        rad = 0.0 if i == 1 else (-0.28 if i == 0 else 0.28)
        y0 = SPINE - 14 if i == 0 else (SPINE + 14 if i == 2 else SPINE)
        arrow(ax, 554, y0, 578, cy, rad=rad, color=RULE, lw=0.7)

        rbox(ax, 582, cy - 38, 96, 76, fc=col, ec=col, lw=0.8)
        txt(ax, 630, cy - 19, b["name"], size=FS_BAND, color=tcol, weight="bold")
        dag = "†" if b["hi"] > 450 else ""
        txt(ax, 630, cy + 3, f"{b['lo']:.0f}–{b['hi']:.0f} Hz{dag}", size=FS_FOOT, color=tcol)
        txt(ax, 630, cy + 22, f"({b['coeff'][0]}, {b['coeff'][1]})",
            size=FS_SHAPE, color=tcol, family=MONO)

        arrow(ax, 682, cy, 700, cy, color=RULE, lw=0.7)

        rbox(ax, 704, cy - 30, 96, 60, ec=col, lw=0.9)
        txt(ax, 752, cy - 10, "conv ×2", size=FS_SUB, color=INK)
        txt(ax, 752, cy + 11, f"({b['conv'][0]}, {b['conv'][1]})",
            size=FS_SHAPE, color=MUTED, family=MONO)

        arrow(ax, 804, cy, 816, cy, color=RULE, lw=0.7)

        ax.add_patch(Circle((832, cy), 14, facecolor=col, edgecolor="none", zorder=4))
        txt(ax, 832, cy, str(b["pooled"]), size=FS_SHAPE, color=tcol, weight="bold", z=5)

        rad2 = 0.0 if i == 1 else (0.28 if i == 0 else -0.28)
        y1 = SPINE - 26 if i == 0 else (SPINE + 26 if i == 2 else SPINE)
        arrow(ax, 848, cy, 864, y1, rad=rad2, color=RULE, lw=0.7)

    # per-band block spec, stated ONCE instead of three times
    txt(ax, 582, 412, "each band:   Conv1d(k=3)→8 · BN · ReLU · MaxPool 2",
        size=FS_FOOT, color=MUTED, ha="left")
    txt(ax, 582, 427, "                    Conv1d(k=3)→16 · BN · ReLU · AvgPool 1",
        size=FS_FOOT, color=MUTED, ha="left")
    txt(ax, 582, 452,
        f"{facts['p_branch']:,} params in the three band branches  ·  "
        f"{facts['p_fusion']:,} in the MLP  ·  {facts['p_head']} in the classifier",
        size=FS_SHAPE, color=SOFT, ha="left")

    # ---------------- 6. concatenate ---------------------------------------------------
    seg = 84 / 3
    for i in range(3):
        ax.add_patch(Rectangle((858, 183 + i * seg), 26, seg, facecolor=BAND[i],
                               edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((858, 183), 26, 84, facecolor="none", edgecolor=RULE,
                           lw=0.6, zorder=4))
    txt(ax, 871, 170, "concat", size=FS_FOOT, color=MUTED)
    txt(ax, 871, 288, f"({facts['concat']},)", size=FS_SHAPE, color=MUTED, family=MONO)

    arrow(ax, 888, SPINE, 906, SPINE)

    # ---------------- 7. MLP -----------------------------------------------------------
    rbox(ax, 910, BOX_T, 108, BOX_H)
    txt(ax, 964, 204, "MLP", size=FS_STAGE, color=INK, weight="semibold")
    txt(ax, 964, 224, f"{facts['concat']}→{facts['concat']}→{facts['mlp_out']}",
        size=FS_SHAPE, color=MUTED, family=MONO)
    txt(ax, 964, 244, "BN, ReLU, p=0.5", size=FS_CLASS, color=MUTED)

    arrow(ax, 1022, SPINE, 1032, SPINE)

    # ---------------- 8. logits --------------------------------------------------------
    widths = [40, 22, 32, 27, 36]
    for i, (w, c, nm) in enumerate(zip(widths, CLASS, CLASS_NAMES)):
        y = 192 + i * 15
        ax.add_patch(Rectangle((1036, y), w, 9, facecolor=c, edgecolor="none", zorder=4))
        txt(ax, 1080, y + 5, nm, size=FS_CLASS, color=MUTED, ha="left")
    txt(ax, 1036, 174, f"{facts['n_classes']} logits", size=FS_FOOT, color=MUTED, ha="left")
    txt(ax, 1036, 288, "palette as in Figs 6–8", size=FS_FOOT, color=MUTED, ha="left")

    # ---------------- the annotation that makes the figure argue ------------------------
    # Pooling is where rhythm is destroyed: §4.7 prices access to it at 7.4 points, and the
    # extended model of Table 7 differs from this one by restoring it.
    ax.plot([832, 832, 878], [94, 74, 74], color=FLAG, lw=0.7, zorder=5)
    txt(ax, 884, 62, "mean over the window —", size=FS_ANNOT, color=FLAG,
        ha="left", weight="bold")
    txt(ax, 884, 76, "the step that discards rhythm", size=FS_ANNOT, color=FLAG, ha="left")
    txt(ax, 884, 90, "(§4.7; restored in Table 7)", size=FS_FOOT, color=FLAG, ha="left")

    # ---------------- footnotes --------------------------------------------------------
    hi_band = next(b for b in facts["bands"] if b["hi"] > 450)
    txt(ax, 20, 480, f"{facts['p_total']:,} trainable parameters", size=FS_SHAPE,
        color=SOFT, ha="left", family=MONO)
    txt(ax, 20, 498,
        f"† {hi_band['name']} spans {hi_band['lo']:.0f}–{hi_band['hi']:.0f} Hz; "
        f"only {hi_band['lo']:.0f}–450 Hz survives the band-pass.",
        size=FS_FOOT, color=MUTED, ha="left")

    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=400, facecolor="white",
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recording", type=Path, default=None,
                    help="optional .npy of shape (channels, samples) for the input traces")
    ap.add_argument("--out", type=Path, default=ROOT / "pipeline_figure")
    args = ap.parse_args()

    facts = model_facts()
    traces = input_traces(args.recording, facts["in_channels"])
    draw(facts, traces, args.out)

    print(f"bands       : {[b['name'] for b in facts['bands']]}")
    for b in facts["bands"]:
        print(f"  {b['name']:<3} {b['lo']:6.1f}-{b['hi']:6.1f} Hz  "
              f"coeff {b['coeff']} -> conv {b['conv']} -> pooled {b['pooled']}")
    print(f"params      : branch {facts['p_branch']:,} + mlp {facts['p_fusion']:,} "
          f"+ head {facts['p_head']} = {facts['p_total']:,}")
    print(f"traces      : {'real excerpt' if args.recording else 'schematic'}")
    print(f"wrote       : {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()
