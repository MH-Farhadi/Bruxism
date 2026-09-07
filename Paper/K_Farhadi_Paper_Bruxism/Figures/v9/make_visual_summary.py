"""Visual Summary slide for the OJEMB submission (page 1 of main_v9.pdf), 4:3 so that it stays
legible when scaled to the letter page. Every number is quoted from main_v9 or recomputed from
the same prediction ledger as Fig. 3 (seed-0 recall matrix). Art is taken from the paper's own
figures: Paper/Picture1.png (acquisition chain, outcome icons) and Fig. 2(b) (corrected chain
over the measured spectrum); the model stack mirrors Fig. 1(b) in the diagram's colours."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageDraw
import textwrap

P = "/home/kye/Desktop/Depo/Code/Bruxism/Paper/"
FIG = P + "K_Farhadi_Paper_Bruxism/Figures/"
OUT = FIG + "v9/visual_summary"

# ----------------------------------------------------------------- style
import glob
for _f in glob.glob("/usr/share/fonts/opentype/inter/Inter-*.otf"):
    if not _f.endswith("Italic.otf"):
        font_manager.fontManager.addfont(_f)
FAMILY = "Inter" if any(f.name == "Inter" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
INK, INK2, MUTED = "#161a1f", "#4a5361", "#8a93a0"
EDGE, PANEL, SURF = "#d8dce2", "#f5f6f8", "#ffffff"
BLUE, ORANGE = "#2d6fcf", "#e06a2b"
plt.rcParams.update({"font.family": FAMILY, "text.color": INK, "axes.edgecolor": EDGE,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
W, H = 10.0, 7.5
fig = plt.figure(figsize=(W, H), facecolor=SURF)
fx, fy = (lambda x: x / W), (lambda y: y / H)
def ax_in(x, y, w, h, **kw):
    ax = fig.add_axes([fx(x), fy(y), fx(w), fy(h)], **kw); ax.set_zorder(2); return ax
def text(x, y, s, size, color=INK, weight="normal", ha="left", va="baseline", ls=1.25, **kw):
    return fig.text(fx(x), fy(y), s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, linespacing=ls, **kw)
def panel(x, y, w, h, fc=PANEL, ec=EDGE, r=0.08, lw=0.8):
    fig.add_artist(FancyBboxPatch((fx(x), fy(y)), fx(w), fy(h), transform=fig.transFigure,
                                  boxstyle=f"round,pad=0,rounding_size={fx(r)}", fc=fc, ec=ec, lw=lw, mutation_aspect=W / H, zorder=0))
def arrow(x0, x1, y):
    fig.add_artist(FancyArrowPatch((fx(x0), fy(y)), (fx(x1), fy(y)), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1.1, shrinkA=0, shrinkB=0))
def image(x, y, w, h, im, frame=False):
    ax = ax_in(x, y, w, h); ax.imshow(im, interpolation="lanczos"); ax.axis("off")
    if frame:
        for s in ax.spines.values(): s.set_visible(True); s.set_edgecolor(EDGE); s.set_linewidth(0.6)
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    return ax
def fit(x, y, w, h, im):
    """Place im inside the (x, y, w, h) box, preserving aspect, centred."""
    iw, ih = im.size; s = min(w / iw, h / ih); ww, hh = iw * s, ih * s
    return x + (w - ww) / 2, y + (h - hh) / 2, ww, hh

# ----------------------------------------------------------------- art from the paper's figures
pic1 = Image.open(P + "Picture1.png").convert("RGB")
acq = pic1.crop((30, 90, 1729, 1965))                       # head, leads, amplifier, ADC, DSP (nothing cut)
strip = pic1.crop((1880, 1470, 3790, 1912))                  # the five outcome icons with their labels
tiles = [strip.crop((a, 0, b, strip.height)) for a, b in ((77, 324), (420, 712), (755, 1108), (1160, 1438), (1551, 1780))]
gap = 70
rows = [tiles[:3], tiles[3:]]
gw = max(sum(t.width for t in r) + gap * (len(r) - 1) for r in rows)
grid = Image.new("RGB", (gw, strip.height * 2 + 60), "white")
for ri, r in enumerate(rows):
    rw = sum(t.width for t in r) + gap * (len(r) - 1); x = (gw - rw) // 2
    for t in r:
        grid.paste(t, (x, ri * (strip.height + 60))); x += t.width + gap
spec = Image.open(FIG + "filter_defect_and_correction.png").convert("RGB").crop((1301, 100, 2245, 1012))
# The panel's own legend sits bottom-left over a region that holds only the uniform fill and vertical
# lines; replace it by tiling the rows just above it, so the fill and the notch lines run through.
_a = np.asarray(spec).copy(); _y0, _y1, _x0, _x1 = 770, 906, 0, 620
_block = _a[_y0 - 120:_y0, _x0:_x1]
for _yy in range(_y0, _y1):
    _a[_yy, _x0:_x1] = _block[(_yy - _y0) % _block.shape[0]]
spec = Image.fromarray(_a)

# ----------------------------------------------------------------- header
text(0.4, 7.0, "Signal Quality, Representation, and Pre-Modelling Choices", 18.5, weight="bold")
text(0.4, 6.66, "A measured performance budget for surface-EMG jaw-activity classification on unseen participants", 11.5, INK2)
text(0.4, 6.40, "5 adults with diagnosed bruxism  ·  100 one-minute recordings  ·  6,173 one-second windows  ·  "
     "quiet rest and 4 instructed jaw activities  ·  nested leave-one-subject-out validation", 8.0, MUTED)
fig.add_artist(Line2D([fx(0.4), fx(9.6)], [fy(6.24), fy(6.24)], transform=fig.transFigure, color=EDGE, lw=0.8))

# ----------------------------------------------------------------- row A: the pipeline
panel(0.4, 3.3, 9.2, 2.82)
text(0.55, 5.9, "THE PIPELINE, FROM SKIN TO DECISION", 8.2, MUTED, weight="bold")
band_y, band_h = 4.02, 1.78
cap_lead, cap_body = 3.93, 3.79
def caption(cx, lead, body, width_chars):
    text(cx, cap_lead, lead, 8.8, weight="bold", ha="center")
    text(cx, cap_body, body, 6.9, INK2, ha="center", va="top", ls=1.3)

# E1 acquisition
x, y, w, h = fit(0.55, band_y, 1.9, band_h, acq); image(x, y, w, h, acq)
caption(1.5, "Record", "4 bipolar sEMG channels over\ntemporalis and masseter, 1200 Hz,\narchived exactly as acquired", 34)
arrow(2.52, 2.74, band_y + band_h / 2)
# E2 audit and filter
x, y, w, h = fit(2.8, band_y + 0.08, 1.62, band_h - 0.16, spec); image(x, y, w, h, spec, frame=True)
caption(3.61, "Audit and filter", "Filter response (blue) over the\nmeasured raw spectrum (gray): 60 Hz\nwas already gone, 7 harmonics were not", 33)
arrow(4.5, 4.72, band_y + band_h / 2)
# E3 represent and classify: the model stack in Fig. 1(b)'s colours
stack = [("1.0-s windows, 0.5-s stride, z-scored with\ntraining-participant statistics", "#dff1e0", "#4caf50"),
         ("db4 wavelet, 4 levels  →  A4 (<38 Hz),\nD3 (75–150 Hz), D1 (300–450 Hz)", "#fff3c4", "#e0b400"),
         ("Two 3-tap conv blocks per band, averaged\nover time  →  3 × 16 features", "#fbe9d4", "#d9a066"),
         ("MLP 48 → 48 → 32  →  five logits\n5,901 trainable parameters", "#f9d9ec", "#d65db1")]
sx, sw, sh, sg = 4.8, 2.4, 0.375, 0.09
sy = band_y + band_h - sh
for i, (label, fc, ec) in enumerate(stack):
    yy = sy - i * (sh + sg)
    panel(sx, yy, sw, sh, fc=fc, ec=ec, r=0.05, lw=0.9)
    text(sx + sw / 2, yy + sh / 2, label, 6.7, ha="center", va="center", ls=1.25)
    if i < len(stack) - 1:
        fig.add_artist(FancyArrowPatch((fx(sx + sw / 2), fy(yy)), (fx(sx + sw / 2), fy(yy - sg + 0.005)), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=6, color=MUTED, lw=0.8, shrinkA=0, shrinkB=0))
caption(6.0, "Represent and classify", "One small convolutional branch per\nwavelet band, then a shared MLP;\nnested LOSO, three seeds", 50)
arrow(7.3, 7.52, band_y + band_h / 2)
# E4 decide
x, y, w, h = fit(7.6, band_y + 0.05, 1.85, band_h - 0.1, grid); image(x, y, w, h, grid)
caption(8.52, "Decide", "Quiet rest or one of four instructed\njaw activities, every 0.5 s", 38)

# ----------------------------------------------------------------- row B left: the budget
panel(0.4, 0.95, 5.35, 2.2)
text(0.55, 2.95, "WHAT EACH CHOICE WAS WORTH", 8.2, MUTED, weight="bold")
text(0.55, 2.79, "Change in participant-level macro-F1 (points); each stage varied on identical windows, folds and labels",
     6.6, INK2)
rows = [("Signal quality", "60-Hz notch → harmonic notch bank", 29.3),
        ("Normalisation scope †", "strict → per-participant", 9.1),
        ("Learned representation", "band energies → wavelet CNN", 8.1),
        ("Temporal context †", "1.0 s → 5.5 s within a trial", 7.4),
        ("Rhythm representation", "compact → extended wavelet CNN", 2.8),
        ("Capacity on the raw signal", "3.2× parameters, no wavelets", -3.3)]
axB = ax_in(2.75, 1.22, 2.8, 1.48)
ypos = np.arange(len(rows))[::-1]; vals = [r[2] for r in rows]
axB.barh(ypos, vals, height=0.6, color=[BLUE if v > 0 else ORANGE for v in vals], zorder=3)
axB.axvline(0, color=INK2, lw=0.8, zorder=4)
for yv, v in zip(ypos, vals):
    axB.text(v + (0.6 if v > 0 else -0.6), yv, f"{v:+.1f}", va="center", ha="left" if v > 0 else "right",
             fontsize=7.6, fontweight="bold", color=INK)
axB.set_xlim(-7.5, 35); axB.set_ylim(-0.5, len(rows) - 0.5); axB.set_xticks([]); axB.set_yticks([])
for s in axB.spines.values(): s.set_visible(False)
row_h = 1.48 / len(rows)
for i, (name, change, _) in enumerate(rows):
    yc = 1.22 + 1.48 - (i + 0.5) * row_h
    text(0.55, yc + 0.035, name, 7.2, weight="bold", va="center")
    text(0.55, yc - 0.085, change, 6.2, INK2, va="center")
text(0.55, 1.08, "† Legitimate but used by no reported number: presumes a per-user fitting session, or aggregates within a "
     "single-condition trial.\nRows are not additive; three are screening measurements, reliable in direction and optimistic in level.",
     5.6, MUTED, va="center", ls=1.3)

# ----------------------------------------------------------------- row B right: held-out result
panel(5.9, 0.95, 3.7, 2.2)
text(6.05, 2.95, "ON PARTICIPANTS THE MODEL NEVER SAW", 8.2, MUTED, weight="bold")
for cx, num, lab in ((6.62, "81.7%", "accuracy"), (7.75, "72.0%", "macro-F1"), (8.88, "0.958", "macro AUC")):
    text(cx, 2.56, num, 16.5, BLUE, weight="bold", ha="center")
    text(cx, 2.41, lab, 6.8, INK2, ha="center")
text(7.75, 2.27, "means over five held-out participants and three seeds; 5,901 parameters", 6.0, MUTED, ha="center")
recall = np.array([[0.905, 0.017, 0.003, 0.073, 0.002],
                   [0.096, 0.727, 0.144, 0.000, 0.033],
                   [0.013, 0.099, 0.678, 0.199, 0.011],
                   [0.074, 0.007, 0.161, 0.722, 0.037],
                   [0.002, 0.102, 0.007, 0.012, 0.877]])     # seed-0 models, pooled over the five folds (Fig. 3b)
labels = ["Rest", "Movement", "Clenching", "Grinding", "Chewing"]
axM = ax_in(6.55, 1.33, 0.92, 0.9)
axM.imshow(recall, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
for i in range(5):
    for j in range(5):
        axM.text(j, i, f"{recall[i, j]:.2f}", ha="center", va="center", fontsize=4.5,
                 color="white" if recall[i, j] > 0.55 else INK)
axM.set_xticks(range(5)); axM.set_yticks(range(5))
axM.set_xticklabels(labels, rotation=40, ha="right", fontsize=5.4); axM.set_yticklabels(labels, fontsize=5.4)
axM.tick_params(length=0, pad=1.5)
axM.set_xlabel("predicted", fontsize=5.6, labelpad=1); axM.set_ylabel("true", fontsize=5.6, labelpad=1)
for s in axM.spines.values(): s.set_edgecolor(EDGE)
text(7.7, 2.13, "Recall per class, one model per fold.", 6.8, weight="bold", va="top")
text(7.7, 1.97, "Quiet rest is recognised in 91% of its\nwindows. Clenching and grinding are the\nhard pair: 20% and 16% cross over.\n"
     "Movement, the smallest class, absorbs\nchewing windows, so its F1 (46%) is a\nprecision problem, not a recall one.", 6.2, INK2, va="top", ls=1.3)

# ----------------------------------------------------------------- footer: three takeaways
tiles_txt = [("Signal quality set the ceiling",
              "The best model on the uncorrected signal (43.5% macro-F1) scored below the weakest baseline on the "
              "corrected one (61.3%). Correcting the filter chain was worth 29.3 points."),
             ("Representation mattered more than size",
              "Both wavelet models placed above both raw-signal models; 3.2× the parameters on the raw signal lost 3.3 "
              "points. Temporal context and normalisation scope were worth as much as the architecture or more."),
             ("What transfers is the procedure",
              "Audit the signal against what the analysis assumes, publish the interference statistic beside the "
              "accuracy, and price the choices made before modelling against the model.")]
tw, tg = 3.0, 0.1
for i, (lead, body) in enumerate(tiles_txt):
    x0 = 0.4 + i * (tw + tg)
    panel(x0, 0.12, tw, 0.72, fc=SURF)
    text(x0 + 0.12, 0.69, lead, 8.0, weight="bold")
    text(x0 + 0.12, 0.60, textwrap.fill(body, 66), 6.0, INK2, va="top", ls=1.3)

for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=300, facecolor=SURF)
print("saved", OUT, "font:", FAMILY)
