#!/usr/bin/env python3
"""Draw the study overview figure: participant -> sensors -> acquisition -> data handling -> outputs.

This is the system-level companion of the model-level diagram (Figures/Pipeline_V2.png). It
shows, on one page, where the recordings come from, every stage the data pass through
offline, which stages can refuse to proceed, and what the two outputs of the pipeline are
for: a measured budget for the researcher and, as intended use, an activity profile for the
patient and clinician.

Every number drawn here is a value that stands in the manuscript (main_v8.tex): recording
counts and durations, window counts, guard widths, residual-mains shares, parameter counts,
latency and the six rows of the "what determines performance" table. They are literals on
purpose -- this figure is documentation of the study, not a measurement -- and FACTS below
is the single place they live so they can be checked against the text.

Sizing: drawn at 6.9 in wide, the width of a two-column sn-jnl figure*, so point sizes below
are the sizes that reach the page when included at width=\\linewidth. After drawing, every
text element is measured (ink extent, verified against a rendered bitmap) against the width
it was given; anything that overflows is shrunk by a few percent and, if it still overflows,
reported, so layout problems are caught without eyeballing the PNG.

Usage
-----
    python make_study_pipeline_figure.py            # writes Figures/study_pipeline.{pdf,png}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3 (publisher requirement)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "Paper" / "K_Farhadi_Paper_Bruxism" / "Figures"

# --------------------------------------------------------------------------------------
# Numbers that appear in the figure (manuscript values, main_v8.tex).
# --------------------------------------------------------------------------------------
FACTS = dict(
    participants=5, recordings=100, recs_per_participant=20, rec_seconds=60, hours="1.66 h",
    fs="1200 Hz", channels=4, windows="6,173", window="1.0 s", stride="0.5 s",
    guard="0.25 s", startup="0.5 s", outer=5, inner=4, seeds=3, params="5,901",
    latency="1.02 ms", predictions="18,519",
    model_fig="Fig. 5", budget_table="Table 9",
    # "What determines performance": participant-level macro-F1, points
    budget=[("signal quality", 29.3), ("normalisation scope", 9.1),
            ("learned representation", 8.1), ("temporal context", 7.4),
            ("rhythm representation", 2.8), ("capacity, raw signal", -3.3)],
    # recordings per class per participant: rest 1, movement 3, clenching 4, grinding 3, chewing 9
    protocol=[("rest", 1), ("movement", 3), ("clenching", 4), ("grinding", 3), ("chewing", 9)],
)

# --------------------------------------------------------------------------------------
# Palette. Class hues follow the seaborn "colorblind" order used by the results figures.
# --------------------------------------------------------------------------------------
INK = "#1A1C26"
SOFT = "#43475A"
MUTED = "#767A8C"
RULE = "#B9BCCC"
TINT = "#EEF0F8"
STAGE = "#2F3B6B"
FLAG = "#8A5320"
CLASS = {"rest": "#0173B2", "movement": "#DE8F05", "clenching": "#029E73",
         "grinding": "#CC78BC", "chewing": "#D55E00"}

SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

FS_EYEBROW = 6.8
FS_TITLE = 7.2
FS_SUB = 6.3
FS_BODY = 5.8
FS_MONO = 5.6

W, H = 1180, 600          # drawing grid; y is inverted so coordinates read top-down
_TEXTS: list[tuple] = []  # (text artist, max width in grid units, label) for the fit check


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def rbox(ax, x, y, w, h, *, fc="none", ec=RULE, lw=0.8, r=5.0, z=2):
    ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r,
                                boxstyle=f"round,pad={r},rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def arrow(ax, x1, y1, x2, y2, *, color=MUTED, lw=0.8, rad=0.0, ms=5.0, z=3):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw, shrinkA=0,
                                shrinkB=0, mutation_scale=ms,
                                connectionstyle=f"arc3,rad={rad}"), zorder=z)


def seg(ax, p, q, **kw):
    ax.plot([p[0], q[0]], [p[1], q[1]], **kw)


def txt(ax, x, y, s, *, size=FS_BODY, color=SOFT, ha="left", va="center", weight="normal",
        family=SANS, z=5, style="normal", maxw=None):
    t = ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight,
                fontfamily=family, zorder=z, fontstyle=style)
    if maxw is not None:
        _TEXTS.append((t, maxw, s[:44]))
    return t


def guard(ax, x, y, r=5.6):
    """The 'refuses rather than warns' mark: a circle with a diagonal bar, in the flag colour."""
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=FLAG, linewidth=0.9, zorder=6))
    d = r * 0.6
    ax.plot([x - d, x + d], [y + d, y - d], color=FLAG, lw=0.9, zorder=7, solid_capstyle="round")


def seated_participant(ax, ox, oy, s=1.0):
    """Seated silhouette in profile, facing right, with the electrode pairs and the button.

    Returns the anchor points (grid coordinates) of the EMG cable and of the trigger lead.
    """
    def P(x, y):
        return (ox + x * s, oy + y * s)

    pt = 0.421 * s                      # one local unit in points (6.9 in / 1180 units)
    body = dict(color=SOFT, lw=8.5 * pt, solid_capstyle="round", zorder=4)
    chair = dict(color=RULE, lw=3.2 * pt, solid_capstyle="round", zorder=2)
    # chair first, so the figure sits in front of it
    seg(ax, P(40, 76), P(40, 142), **chair)          # backrest
    seg(ax, P(40, 142), P(102, 142), **chair)        # seat
    seg(ax, P(46, 142), P(46, 188), **chair)         # rear leg
    seg(ax, P(96, 142), P(96, 188), **chair)         # front leg
    # figure: head, torso, thigh, shin, upper arm holding the button
    ax.add_patch(Circle(P(72, 58), 17 * s, facecolor=SOFT, edgecolor="none", zorder=4))
    seg(ax, P(68, 78), P(60, 136), **body)           # torso
    seg(ax, P(60, 136), P(94, 136), **body)          # thigh
    seg(ax, P(94, 136), P(94, 180), **body)          # shin
    seg(ax, P(66, 92), P(92, 114), **{**body, "lw": 6.0 * pt})   # arm
    # hand-held trigger button
    ax.add_patch(FancyBboxPatch(P(90, 108), 12 * s, 12 * s,
                                boxstyle="round,pad=0,rounding_size=2",
                                facecolor="white", edgecolor=SOFT, linewidth=0.7, zorder=5))
    ax.add_patch(Circle(P(96, 114), 2.4 * s, facecolor=FLAG, edgecolor="none", zorder=6))
    # electrodes: a temporalis pair on the temple (upper front of the head) and a masseter
    # pair over the angle of the jaw (lower front); both sit inside the head outline
    for (x, y) in ((81, 47), (86, 53), (80, 66), (84, 70)):
        ax.add_patch(Circle(P(x, y), 2.3 * s, facecolor="white", edgecolor=STAGE,
                            linewidth=0.8, zorder=6))
    # short leads running back over the head to one cable
    seg(ax, P(83, 50), P(66, 41), color="white", lw=0.5, zorder=5)
    seg(ax, P(82, 68), P(66, 41), color="white", lw=0.5, zorder=5)
    return P(66, 41), P(102, 114)


def fit_check(fig, ax):
    """Measure every registered text against its allotted width; shrink a little, else report."""
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def width(t):
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=renderer)
        (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return abs(x1 - x0)

    problems = 0
    for t, maxw, label in _TEXTS:
        w = width(t)
        shrinks = 0
        while w > maxw and shrinks < 3:
            t.set_fontsize(t.get_fontsize() * 0.96)
            shrinks += 1
            w = width(t)
        if w > maxw:
            problems += 1
            print(f"  OVERFLOW {w:6.1f} > {maxw:5.1f}  '{label}'")
        elif shrinks:
            print(f"  shrunk x{shrinks} ({t.get_fontsize():.1f} pt)  '{label}'")
    print(f"fit check: {len(_TEXTS)} texts, {problems} overflow(s)")


# --------------------------------------------------------------------------------------
# the drawing
# --------------------------------------------------------------------------------------
def draw(out_stem: Path) -> None:
    F = FACTS
    fig, ax = plt.subplots(figsize=(6.9, 6.9 * H / W))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    AX0, AX1 = 20, 395      # column A
    BX0, BX1 = 415, 848     # column B
    CX0, CX1 = 868, 1160    # column C
    TOP = 44
    LINE = 11.5

    # ---------------------------------------------------------------- eyebrows -------
    txt(ax, AX0, 16, "A   COLLECTION", size=FS_EYEBROW, color=MUTED, weight="bold")
    txt(ax, BX0, 16, "B   DATA HANDLING", size=FS_EYEBROW, color=MUTED, weight="bold")
    txt(ax, CX0, 16, "C   OUTPUTS", size=FS_EYEBROW, color=MUTED, weight="bold")
    ax.plot([AX0, CX1], [28, 28], color=RULE, lw=0.6, zorder=1)

    # ================================================================ A: COLLECTION ==
    cable, lead = seated_participant(ax, 18, TOP - 2, s=0.9)

    tx = 130
    tw = AX1 - tx
    y = TOP + 8
    txt(ax, tx, y, "Participant", size=FS_TITLE, color=INK, weight="semibold", maxw=tw)
    txt(ax, tx, y + LINE + 2, f"seated, awake; {F['participants']} adults with a prior",
        maxw=tw)
    txt(ax, tx, y + 2 * LINE + 2, "clinical diagnosis of bruxism", maxw=tw)

    y = TOP + 50
    txt(ax, tx, y, "Bipolar surface-EMG pairs", size=FS_SUB, color=STAGE, weight="semibold",
        maxw=tw)
    txt(ax, tx, y + LINE + 1, "temporalis and masseter, both sides:", maxw=tw)
    txt(ax, tx, y + 2 * LINE + 1, f"{F['channels']} differential channels", maxw=tw)

    y = TOP + 94
    txt(ax, tx, y, "Hand-held button", size=FS_SUB, color=FLAG, weight="semibold", maxw=tw)
    txt(ax, tx, y + LINE + 1, "pressed while a task is performed:", maxw=tw)
    txt(ax, tx, y + 2 * LINE + 1, "the only source of the class labels", maxw=tw)

    y = TOP + 138
    txt(ax, tx, y, "Camera and microphone", size=FS_SUB, color=MUTED, weight="semibold",
        maxw=tw)
    txt(ax, tx, y + LINE + 1, "recorded for review; not analysed", maxw=tw)

    # amplifier / ADC
    ay = TOP + 178
    aw = AX1 - AX0
    rbox(ax, AX0, ay, aw, 50, fc=TINT, ec=RULE)
    txt(ax, AX0 + 12, ay + 13, "Amplifier and ADC", size=FS_TITLE, color=INK, weight="semibold",
        maxw=aw - 24)
    txt(ax, AX0 + 12, ay + 27, f"hardware notch at 60 Hz only; {F['fs']}; ADC units",
        maxw=aw - 24)
    txt(ax, AX0 + 12, ay + 39, "electrode and amplifier models undocumented", color=MUTED,
        style="italic", maxw=aw - 24)
    # cable and trigger lead into the amplifier
    ax.plot([cable[0], 32, 32], [cable[1], cable[1], ay], color=STAGE, lw=0.8, zorder=3)
    ax.plot([lead[0], 122, 122], [lead[1], lead[1], ay], color=FLAG, lw=0.8, zorder=3)
    txt(ax, 36, ay - 7, f"EMG × {F['channels']}", size=5.2, color=STAGE, maxw=60)
    txt(ax, 126, ay - 7, "trigger", size=5.2, color=FLAG, maxw=60)
    arrow(ax, AX0 + aw / 2, ay + 50, AX0 + aw / 2, ay + 62, color=MUTED, lw=0.8)

    # acquisition PC
    py = ay + 62
    rbox(ax, AX0, py, aw, 72, fc="white", ec=RULE)
    txt(ax, AX0 + 12, py + 13, "Acquisition PC", size=FS_TITLE, color=INK, weight="semibold",
        maxw=aw - 24)
    txt(ax, AX0 + 12, py + 27, f"each: CSV ({F['channels']} EMG, trigger, mic), metadata, video", maxw=aw - 24)
    txt(ax, AX0 + 12, py + 44, f"{F['recs_per_participant']} recordings × "
        f"{F['rec_seconds']} s per participant", size=6.0, color=STAGE, weight="semibold",
        maxw=aw - 24)
    txt(ax, AX0 + 12, py + 57, f"{F['recordings']} recordings, {F['hours']} in total",
        size=6.0, color=STAGE, weight="semibold", maxw=aw - 24)

    # session protocol
    sy = py + 90
    txt(ax, AX0, sy, "Session protocol", size=FS_TITLE, color=INK, weight="semibold", maxw=160)
    txt(ax, AX0 + 164, sy + 1, "recordings per participant", color=MUTED, maxw=aw - 164)
    by, bh = sy + 12, 13
    total = sum(n for _, n in F["protocol"])
    x = AX0
    for name, n in F["protocol"]:
        w = aw * n / total
        ax.add_patch(Rectangle((x, by), w - 1.2, bh, facecolor=CLASS[name], edgecolor="none",
                               zorder=4))
        txt(ax, x + (w - 1.2) / 2, by + bh / 2 + 0.5, f"{n}", size=5.6, color="white",
            ha="center", family=MONO, z=6)
        x += w
    legend = [("rest", "quiet, 1 min"),
              ("movement", "open/close, deviation, protrusion"),
              ("clenching", "bite L/R, molar, incisor; 3 s on, 3 s off"),
              ("grinding", "instructed"),
              ("chewing", "cheese, carrots, gum")]
    ly = by + bh + 11
    for i, (name, desc) in enumerate(legend):
        yy = ly + i * LINE
        ax.add_patch(Rectangle((AX0, yy - 3.2), 6.5, 6.5, facecolor=CLASS[name],
                               edgecolor="none", zorder=4))
        txt(ax, AX0 + 10, yy, name, color=INK, weight="semibold", maxw=86)
        txt(ax, AX0 + 100, yy, desc, maxw=aw - 100)
    txt(ax, AX0, ly + 5 * LINE + 1, "each recording is 1 min and holds one condition",
        color=MUTED, maxw=aw)

    # trigger trace with the guarded edges
    ty = ly + 6 * LINE + 10
    txt(ax, AX0, ty, "trigger", maxw=52)
    tx0, tx1 = AX0 + 56, AX1
    t_ = np.linspace(0, 1, 600)
    runs = [(0.05, 0.22), (0.30, 0.42), (0.52, 0.82), (0.88, 0.97)]
    sig = np.zeros_like(t_)
    for a, b in runs:
        sig[(t_ >= a) & (t_ <= b)] = 1
    ax.plot(tx0 + t_ * (tx1 - tx0), ty + 4 - sig * 8, color=SOFT, lw=0.8, zorder=4,
            drawstyle="steps-post")
    a, b = runs[2]
    for edge in (tx0 + a * (tx1 - tx0), tx0 + b * (tx1 - tx0) - 6):
        ax.add_patch(Rectangle((edge, ty - 7), 6, 14, facecolor=FLAG, alpha=0.3,
                               edgecolor="none", zorder=3))
    txt(ax, AX0, ty + 17, "windows are cut only inside marked intervals;", color=MUTED,
        maxw=aw)
    txt(ax, AX0, ty + 17 + LINE, f"guards: {F['guard']} per edge (shaded), {F['startup']} per start", color=MUTED, maxw=aw)
    a_bottom = ty + 17 + LINE + 8

    # collection -> handling: the recordings enter stage 1
    ax.plot([AX1, 405, 405], [py + 36, py + 36, TOP + 31], color=MUTED, lw=0.9, zorder=3,
            solid_joinstyle="round")
    arrow(ax, 405, TOP + 31, BX0, TOP + 31, color=MUTED, lw=0.9)

    # ============================================================ B: DATA HANDLING ====
    MEAS_W = 112
    body_x = BX0 + 26
    body_w = BX1 - 12 - body_x
    gx = BX1 - 12 - MEAS_W - 14
    rows = [
        ("Manifest and signal audit",
         ["schema, channel fingerprints, mains-harmonic share,",
          "startup transients; a duplicated channel is refused"],
         "100/100 unique", True),
        ("Filter chain",
         ["notches 60–420 Hz, band-pass 20–450 Hz, zero-phase,",
          "whole recording; residual mains share before → after"],
         "0.673 → 0.012", False),
        ("Window index",
         [f"{F['window']} windows, {F['stride']} stride, wholly inside one marked",
          "interval; rest only from the dedicated rest recordings"],
         f"{F['windows']} windows", False),
        ("Nested LOSO evaluation",
         [f"leave-one-subject-out, {F['outer']} outer × {F['inner']} inner folds, "
          f"{F['seeds']} seeds;",
          "every statistic fitted on training participants only;",
          "held-out participant ids sealed"],
         f"{F['outer'] * F['seeds']} refits", True),
        ("Representation and model",
         [f"db4 wavelet bands A4, D3, D1 → compact CNN ({F['model_fig']}),",
          f"{F['latency']} per window; a band the filter cut is refused"],
         f"{F['params']} params", True),
        ("Prediction ledger",
         ["each held-out window once per seed, with the config,",
          "manifest, window and checkpoint hashes; every table",
          "and figure in the paper is regenerated from it"],
         f"{F['predictions']} rows", False),
    ]
    y = TOP
    row_mid = []
    for i, (title, body, meas, has_guard) in enumerate(rows):
        h = 36 + LINE * len(body) + 4
        rbox(ax, BX0, y, BX1 - BX0, h, fc=TINT if i % 2 == 0 else "white", ec=RULE)
        txt(ax, BX0 + 11, y + 15, f"{i + 1}", color=MUTED, family=MONO, maxw=14)
        txt(ax, body_x, y + 15, title, size=FS_TITLE, color=INK, weight="semibold",
            maxw=(gx - 10) - body_x)
        for j, line in enumerate(body):
            txt(ax, body_x, y + 31 + j * LINE, line, maxw=body_w)
        txt(ax, BX1 - 12, y + 15, meas, size=FS_MONO, color=STAGE, ha="right", family=MONO,
            maxw=MEAS_W)
        if has_guard:
            guard(ax, gx, y + 15)
        row_mid.append(y + h / 2)
        if i < len(rows) - 1:
            arrow(ax, BX0 + 120, y + h, BX0 + 120, y + h + 16, color=RULE, lw=0.8, ms=4)
        y += h + 16
    b_bottom = y - 16
    bottom = max(a_bottom, b_bottom)

    # handling -> outputs
    arrow(ax, BX1, row_mid[-1], CX0, TOP + 118, color=MUTED, lw=0.9, rad=-0.22)
    arrow(ax, BX1, row_mid[-1], CX0, bottom - 60, color=MUTED, lw=0.9, rad=0.18)

    # ================================================================ C: OUTPUTS ======
    cw = CX1 - CX0
    inner = cw - 24
    top_h = 250
    rbox(ax, CX0, TOP, cw, top_h, fc="white", ec=RULE)
    txt(ax, CX0 + 12, TOP + 13, "For researchers", size=FS_TITLE, color=INK,
        weight="semibold", maxw=inner)
    txt(ax, CX0 + 12, TOP + 27, f"the measured budget ({F['budget_table']})",
        color=MUTED, maxw=inner)
    txt(ax, CX0 + 12, TOP + 38, "Δ macro-F1, participant level, points", color=MUTED,
        maxw=inner)
    base = CX0 + 174
    span = CX1 - 14 - base
    scale = span / 29.3
    for i, (name, v) in enumerate(F["budget"]):
        yy = TOP + 60 + i * 24
        txt(ax, base - 6, yy, name, ha="right", maxw=base - 6 - (CX0 + 10))
        if v >= 0:
            ax.add_patch(Rectangle((base, yy - 5), v * scale, 10, facecolor=STAGE,
                                   edgecolor="none", zorder=4))
            if v * scale > 34:
                txt(ax, base + v * scale - 3, yy, f"+{v:.1f}", size=5.6, color="white",
                    ha="right", family=MONO, z=6)
            else:
                txt(ax, base + v * scale + 3, yy, f"+{v:.1f}", size=5.6, color=STAGE,
                    family=MONO)
        else:
            ax.add_patch(Rectangle((base, yy - 5), -v * scale, 10, facecolor=FLAG,
                                   edgecolor="none", zorder=4))
            txt(ax, base - v * scale + 3, yy, f"−{-v:.1f}", size=5.6, color=FLAG, family=MONO)
    ax.plot([base, base], [TOP + 50, TOP + 60 + 5 * 24 + 8], color=RULE, lw=0.6, zorder=3)
    txt(ax, CX0 + 12, TOP + 210, "not additive; see the table notes", color=MUTED,
        maxw=inner)
    txt(ax, CX0 + 12, TOP + 226, "signal quality sets the ceiling;", style="italic", maxw=inner)
    txt(ax, CX0 + 12, TOP + 226 + LINE, "above it, representation beats size",
        style="italic", maxw=inner)

    # patients and clinicians: the activity profile (intended use)
    oy = TOP + top_h + 14
    rbox(ax, CX0, oy, cw, bottom - oy, fc="white", ec=RULE)
    txt(ax, CX0 + 12, oy + 13, "For patients and clinicians", size=FS_TITLE, color=INK,
        weight="semibold", maxw=inner)
    txt(ax, CX0 + 12, oy + 27, "intended use: an activity profile", color=MUTED, maxw=inner)
    seq = [("rest", 6), ("movement", 2), ("chewing", 9), ("rest", 4), ("clenching", 5),
           ("grinding", 4), ("rest", 5)]
    sx, sw, sy2, sh = CX0 + 12, inner, oy + 38, 13
    n = sum(k for _, k in seq)
    x = sx
    for name, k in seq:
        w = sw * k / n
        ax.add_patch(Rectangle((x, sy2), w - 1.0, sh, facecolor=CLASS[name], edgecolor="none",
                               zorder=4))
        x += w
    for k in range(0, 9):
        xx = sx + sw * k / 8
        ax.plot([xx, xx], [sy2 + sh + 1.5, sy2 + sh + 4.5], color=RULE, lw=0.6, zorder=3)
    txt(ax, sx, sy2 + sh + 12, f"one decision per {F['stride']}, {F['window']} window",
        color=MUTED, maxw=inner)
    for i, name in enumerate(CLASS):
        col, row = i % 3, i // 3
        xx = sx + col * 90
        yy = sy2 + sh + 28 + row * LINE
        ax.add_patch(Rectangle((xx, yy - 3.2), 6.5, 6.5, facecolor=CLASS[name],
                               edgecolor="none", zorder=4))
        txt(ax, xx + 10, yy, name, maxw=76)
    my = sy2 + sh + 28 + 2 * LINE + 12
    txt(ax, CX0 + 12, my, "measured here: rest recalled at 91 %;", maxw=inner)
    txt(ax, CX0 + 12, my + LINE, "7.7 % of rest windows were read as", maxw=inner)
    txt(ax, CX0 + 12, my + 2 * LINE, "clenching or grinding: the closest", maxw=inner)
    txt(ax, CX0 + 12, my + 3 * LINE, "false-alarm analogue this design has", maxw=inner)
    cy = bottom - 12 - 2 * LINE
    txt(ax, CX0 + 12, cy, "instructed laboratory conditions only:", color=FLAG, maxw=inner)
    txt(ax, CX0 + 12, cy + LINE, "not a diagnosis, and not validated", color=FLAG, maxw=inner)
    txt(ax, CX0 + 12, cy + 2 * LINE, "on spontaneous behaviour", color=FLAG, maxw=inner)

    # ---------------------------------------------------------------- footnotes ------
    fy = bottom + 20
    guard(ax, AX0 + 6, fy, r=4.8)
    txt(ax, AX0 + 17, fy, "marks a stage that refuses rather than warns: a duplicated channel, a wavelet "
        "band the filter removed, a held-out participant inside a fitted statistic.", color=MUTED, maxw=CX1 - AX0 - 17)
    txt(ax, AX0, fy + LINE + 1, "Arrows carry data. Every number is a value measured in this "
        "study; the microphone and video were recorded for review and enter no analysis.",
        color=MUTED, maxw=CX1 - AX0)

    fit_check(fig, ax)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=400, facecolor="white", bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print(f"columns end at A={a_bottom:.0f} B={b_bottom:.0f}; footnotes at {fy:.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "study_pipeline")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    draw(args.out)
    print(f"wrote {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()
