#!/usr/bin/env python3
"""Build the scope-update deck and its speaker script.

    python3 Presentation/build_deck.py

Writes, next to this file:
    scope_update_deck.pptx   the deck (speaker notes embedded)
    script.md                the presentation script, generated from the same notes
    assets/                  the figures the deck embeds (copied from Paper/ and advisor/)
                             plus the budget chart drawn here from Table 8 of main_v8.tex

Every number on the slides is quoted from main_v8.tex (26 Aug 2026), the advisor
briefing (advisor/briefing.tex, 19 Aug 2026), cause.md / audio.md (git HEAD) and the
Scientific Reports decision letter (Paper/Reviews). Venue facts (locations, deadlines,
fees, impact factors) were checked on 27-28 Aug 2026; sources are listed in venues.md.
Nothing is recomputed here.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / "assets"
PAPER_FIG = ROOT / "Paper" / "K_Farhadi_Paper_Bruxism" / "Figures"
ADVISOR_FIG = ROOT / "advisor" / "figures"

DECK = HERE / "scope_update_deck.pptx"
SCRIPT = HERE / "script.md"

# ----------------------------------------------------------------------------- palette
INK = RGBColor(0x1A, 0x20, 0x2C)
SLATE = RGBColor(0x4A, 0x55, 0x68)
MUTED = RGBColor(0x89, 0x87, 0x81)
ACCENT = RGBColor(0x1F, 0x6F, 0xEB)
BAD = RGBColor(0xC1, 0x27, 0x2D)
GOOD = RGBColor(0x0B, 0x6E, 0x4F)
WASH = RGBColor(0xF2, 0xF5, 0xF9)
WASH_BAD = RGBColor(0xFB, 0xEF, 0xEF)
WASH_GOOD = RGBColor(0xEC, 0xF6, 0xF2)
SURFACE = RGBColor(0xFC, 0xFC, 0xFB)
RULE = RGBColor(0xE1, 0xE0, 0xD9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Calibri"
SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
LM = Inches(0.6)
CW = SLIDE_W - 2 * LM
DATE = "August 2026"          # meeting date is open; venue facts are dated where quoted
CHECKED = "28 Aug 2026"       # date the venue facts were last verified
FOOTER = f"Bruxism paper · scope update · {DATE}"


# ----------------------------------------------------------------------------- helpers
def _style_run(run, size, color=INK, bold=False, italic=False, font=FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def _bullet(p, char="•", level=0, hang=0.2):
    """Real PowerPoint bullet with a hanging indent (XML; python-pptx has no API)."""
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(hang + 0.25 * level))))
    pPr.set("indent", str(-int(Inches(hang))))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum", "a:buFont", "a:buClr"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    bu_font = etree.SubElement(pPr, qn("a:buFont"))
    bu_font.set("typeface", "Arial")
    bu_char = etree.SubElement(pPr, qn("a:buChar"))
    bu_char.set("char", char)


def add_text(slide, x, y, w, h, items, size=14, color=INK, bold=False, italic=False,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.08,
             space_after=4, inset=0.05):
    """items: str | dict(text, size, color, bold, italic, bullet, level, space_after,
    align, runs=[(text, {style})])."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(inset)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    first = True
    for item in items:
        if isinstance(item, str):
            item = dict(text=item)
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = item.get("align", align)
        p.line_spacing = item.get("line_spacing", line_spacing)
        p.space_after = Pt(item.get("space_after", space_after))
        p.space_before = Pt(item.get("space_before", 0))
        runs = item.get("runs") or [(item.get("text", ""), {})]
        for text, style in runs:
            r = p.add_run()
            r.text = text
            _style_run(r, style.get("size", item.get("size", size)),
                       style.get("color", item.get("color", color)),
                       style.get("bold", item.get("bold", bold)),
                       style.get("italic", item.get("italic", italic)))
        if item.get("bullet"):
            _bullet(p, item.get("char", "•"), item.get("level", 0))
    return tb


def add_rect(slide, x, y, w, h, fill=WASH, line=None, rounded=False, radius=0.06):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, x, y, w, h)
    if rounded:
        shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    return shp


def add_image(slide, path, x, y, w=None, h=None):
    iw, ih = Image.open(path).size
    if w is not None and h is None:
        h = int(w * ih / iw)
    elif h is not None and w is None:
        w = int(h * iw / ih)
    elif w is not None and h is not None:
        s = min(w / iw, h / ih)
        w2, h2 = int(iw * s), int(ih * s)
        x, y = x + (w - w2) // 2, y + (h - h2) // 2
        w, h = w2, h2
    return slide.shapes.add_picture(str(path), x, y, width=w, height=h)


def add_table(slide, x, y, col_widths_in, rows, size=11, row_h=0.3, header=True,
              header_fill=SLATE, col_align=None, bold_first_col=False, zebra=True):
    nrows, ncols = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nrows, ncols, x, y,
                                Inches(sum(col_widths_in)), Inches(row_h * nrows))
    tbl = gf.table
    tbl.first_row = header
    tbl.horz_banding = tbl.vert_banding = tbl.first_col = tbl.last_row = tbl.last_col = False
    for i, cw in enumerate(col_widths_in):
        tbl.columns[i].width = Inches(cw)
    for r in range(nrows):
        tbl.rows[r].height = Inches(row_h)
        for c in range(ncols):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.025)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            is_header = header and r == 0
            lines = str(rows[r][c]).split("\n")
            for li, line in enumerate(lines):
                p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
                if col_align and col_align[c]:
                    p.alignment = col_align[c]
                run = p.add_run()
                run.text = line
                _style_run(run, size, WHITE if is_header else INK,
                           bold=is_header or (bold_first_col and c == 0))
            cell.fill.solid()
            if is_header:
                cell.fill.fore_color.rgb = header_fill
            else:
                cell.fill.fore_color.rgb = WASH if (zebra and r % 2 == 0) else WHITE
    return gf


def stat_tile(slide, x, y, w, h, value, label, value_color=INK, fill=WASH, vsize=26, lsize=11):
    add_rect(slide, x, y, w, h, fill=fill, rounded=True, radius=0.08)
    add_text(slide, x + Inches(0.12), y + Inches(0.06), w - Inches(0.24), Inches(0.6),
             [dict(text=value, size=vsize, bold=True, color=value_color)], space_after=0)
    add_text(slide, x + Inches(0.12), y + Inches(0.62), w - Inches(0.24), h - Inches(0.66),
             [dict(text=label, size=lsize, color=SLATE)], space_after=0, line_spacing=1.02)


def header_line(slide, x, y, w, text, color=ACCENT, size=13):
    add_text(slide, x, y, w, Inches(0.32), [dict(text=text, size=size, bold=True, color=color)],
             space_after=0)


def caption(slide, x, y, w, text, size=10):
    add_text(slide, x, y, w, Inches(0.6), [dict(text=text, size=size, color=MUTED)],
             line_spacing=1.0, space_after=0)


class Deck:
    """Numbers slides in build order so inserting a slide never renumbers by hand."""

    def __init__(self, prs, total):
        self.prs = prs
        self.total = total
        self.n = 0
        self.order: list[str] = []

    def slide(self, key, kicker, title, notes):
        self.n += 1
        self.order.append(key)
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = SURFACE
        add_text(s, LM, Inches(0.3), CW, Inches(0.3),
                 [dict(text=kicker.upper(), size=10.5, bold=True, color=ACCENT)], space_after=0)
        # a title that would wrap at 25 pt collides with the accent rule; shrink it instead
        tsize = 25 if len(title) <= 72 else 21
        add_text(s, LM, Inches(0.55), CW, Inches(0.85),
                 [dict(text=title, size=tsize, bold=True, color=INK)], space_after=0, line_spacing=1.0)
        add_rect(s, LM, Inches(1.36), Inches(0.9), Inches(0.045), fill=ACCENT)
        add_text(s, LM, Inches(7.05), Inches(8), Inches(0.3),
                 [dict(text=FOOTER, size=9, color=MUTED)], space_after=0)
        add_text(s, SLIDE_W - LM - Inches(1.5), Inches(7.05), Inches(1.5), Inches(0.3),
                 [dict(text=f"{self.n} / {self.total}", size=9, color=MUTED, align=PP_ALIGN.RIGHT)],
                 space_after=0)
        s.notes_slide.notes_text_frame.text = notes
        return s

    def title_slide(self, key, notes):
        self.n += 1
        self.order.append(key)
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = SURFACE
        s.notes_slide.notes_text_frame.text = notes
        return s


# ----------------------------------------------------------------------------- chart
BUDGET_ROWS = [
    # choice, change measured, delta (points macro-F1), harness, used in reported results
    ("Signal quality", "60-Hz notch → harmonic notch bank", 29.3, "Confirmatory", "Yes"),
    ("Normalisation scope", "Strict → per-participant", 9.1, "Screening", "No – product decision"),
    ("Learned representation", "Band energies → wavelet CNN", 8.1, "Conf. / screen.", "Yes"),
    ("Temporal context", "1.0 s → 5.5 s within a trial", 7.4, "Screening", "No – trial-level"),
    ("Rhythm representation", "Compact → extended wavelet CNN", 2.8, "Arch. sweep", "Secondary"),
    ("Capacity on the raw signal", "Wavelet CNN → early-fusion CNN (3.2× params)", -3.3, "Arch. sweep", "No"),
]


def make_budget_chart(out: Path):
    """Table 8 of main_v8.tex as a diverging bar chart (dataviz skill palette, validated)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from matplotlib.transforms import blended_transform_factory

    family = "DejaVu Sans"
    try:
        fpath = subprocess.check_output(["fc-match", "-f", "%{file}", "Carlito"]).decode().strip()
        if "carlito" in fpath.lower():
            for p in Path(fpath).parent.glob("Carlito*"):
                fm.fontManager.addfont(str(p))
            family = "Carlito"
    except Exception:
        pass
    plt.rcParams["font.family"] = family

    BLUE, RED = "#2a78d6", "#e34948"        # diverging poles, validated (light surface)
    INKc, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
    GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

    fig_w, fig_h = 10.6, 5.4
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=220, facecolor=SURF)
    ax_box = [0.33, 0.12, 0.44, 0.72]
    ax = fig.add_axes(ax_box)
    ax.set_facecolor(SURF)
    n = len(BUDGET_ROWS)
    ys = list(range(n))[::-1]
    xmin, xmax = -6.5, 34.0
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.6, n - 0.4)
    for gx in (0, 10, 20, 30):
        ax.axvline(gx, color=BASE if gx == 0 else GRID, lw=1.1 if gx == 0 else 0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xticklabels(["0", "+10", "+20", "+30 points"], color=MUT, fontsize=9)
    ax.tick_params(axis="x", length=0, pad=6)

    ax_w_in, ax_h_in = fig_w * ax_box[2], fig_h * ax_box[3]
    dx, dy = xmax - xmin, (n - 0.4) - (-0.6)
    bar_h = 0.40
    r_x = (bar_h / 2) * (ax_h_in / dy) * (dx / ax_w_in)
    aspect = (dy * ax_w_in) / (dx * ax_h_in)
    trans = blended_transform_factory(ax.transAxes, ax.transData)

    for (name, change, val, harness, reported), y in zip(BUDGET_ROWS, ys):
        color = BLUE if val > 0 else RED
        x0, x1 = (0, val) if val > 0 else (val, 0)
        ax.add_patch(FancyBboxPatch((x0, y - bar_h / 2), x1 - x0, bar_h,
                                    boxstyle=f"round,pad=0,rounding_size={r_x}",
                                    mutation_aspect=aspect, fc=color, ec="none", zorder=2))
        if val > 0:
            ax.add_patch(Rectangle((0, y - bar_h / 2), r_x, bar_h, fc=color, ec="none", zorder=3))
            ax.text(val + 0.7, y, f"+{val:.1f}", ha="left", va="center", fontsize=10.5,
                    fontweight="bold", color=INKc)
        else:
            ax.add_patch(Rectangle((-r_x, y - bar_h / 2), r_x, bar_h, fc=color, ec="none", zorder=3))
            ax.text(val - 0.7, y, f"−{abs(val):.1f}", ha="right", va="center", fontsize=10.5,
                    fontweight="bold", color=INKc)
        ax.text(-0.025, y + 0.13, name, transform=trans, ha="right", va="center",
                fontsize=11, fontweight="bold", color=INKc)
        ax.text(-0.025, y - 0.2, change, transform=trans, ha="right", va="center",
                fontsize=8.8, color=SEC)
        ax.text(1.03, y + 0.13, harness, transform=trans, ha="left", va="center",
                fontsize=9, color=SEC)
        ax.text(1.03, y - 0.2, reported, transform=trans, ha="left", va="center",
                fontsize=8.8, color=MUT)

    ax.text(1.03, n - 0.45, "Harness · used in reported results", transform=trans,
            ha="left", va="center", fontsize=8.5, color=MUT, fontstyle="italic")
    fig.text(0.03, 0.955, "Change in participant-level macro-F1 when one stage is changed",
             fontsize=12.5, fontweight="bold", color=INKc)
    fig.text(0.03, 0.905,
             "Same 6,173 windows and five outer folds in every arm. Entries are not additive "
             "and come from different harnesses.", fontsize=9.5, color=SEC)
    lx = 0.80
    for i, (lab, col) in enumerate((("gain", BLUE), ("loss", RED))):
        fig.patches.append(Rectangle((lx + i * 0.075, 0.948), 0.013, 0.022, transform=fig.transFigure,
                                     fc=col, ec="none"))
        fig.text(lx + i * 0.075 + 0.018, 0.959, lab, fontsize=9.5, color=SEC, va="center")
    fig.savefig(out, dpi=220, facecolor=SURF)
    plt.close(fig)


# ----------------------------------------------------------------------------- venue data
# JIF = Clarivate Journal Impact Factor, JCR 2025 (released 17 Jun 2026) unless a year is given.
# Cross-checked 28 Aug 2026 against publisher pages where those rendered (IOP, Wiley, Springer,
# IEEE Sensors Council, IEEE Access) and a year-by-year JCR aggregator for the rest. See venues.md.
JOURNALS = [
    # name, JIF, model/fee, publisher-stated speed, why it fits, read
    ("IEEE J. Biomedical & Health Informatics", "7.7", "hybrid", "n.s.",
     "Sensor-to-decision pipelines with rigorous validation; the EMBS readership", "★ first choice"),
    ("IEEE Trans. Biomedical Engineering", "4.4", "hybrid", "n.s.",
     "Strongest line on a CV; lead with the measurement-validity contribution", "selective, slow"),
    ("Physiological Measurement (IOP / IPEM)", "2.5", "hybrid", "49 d to first decision (median)",
     "Scope is “new methods of measurement and their validation” — the audit framing", "good fit"),
    ("IEEE Open J. Eng. in Medicine & Biology", "3.1", "OA · US$2,160", "n.s. (“fast”)",
     "EMBS-branded, deliberately fast; citable before the grant is written", "speed play"),
    ("IEEE Access", "4.2", "OA · US$2,160", "n.s.",
     "Broad, fast, no page limit (≤ 20 pp advised); less selective", "speed play"),
    ("Computers in Biology and Medicine (Elsevier)", "6.3 (2024)", "OA · US$3,080", "n.s.",
     "Computational-methods audience; pipelines and validation in scope", "good fit"),
    ("Biomedical Signal Processing and Control (Elsevier)", "5.7", "hybrid", "n.s.",
     "Filter / spectrum finding and the wavelet comparison squarely in scope", "strong second"),
    ("J. Electromyography & Kinesiology (Elsevier)", "2.7", "hybrid", "n.s.",
     "EMG-methods readership — where the harmonic-notch finding lands hardest", "if EMG leads"),
]
JOURNALS_ALSO = ("Also considered (JIF 2025): Computer Methods and Programs in Biomedicine 6.4 · "
                 "Medical & Biological Engineering & Computing 3.1 (an MBEC draft exists) · IEEE Sensors "
                 "Journal 4.5 (8.8 wk to e-publication) · Journal of Oral Rehabilitation 3.8 (8 d to first "
                 "decision; strategic — the advisor’s community; needs reframing). Journals take rolling "
                 "submissions: there is no deadline to hit.")

CONFERENCES = [
    # conference, next edition, where, paper deadline, proceedings, verdict
    ("IEEE EMBS BHI 2027", "Nov 2027", "USA, city TBD", "expected Jun 2027 (2026: 12 Jun → 3 Jul)", "IEEE Xplore", "watch — US confirmed, city TBD"),
    ("IEEE EMBS BSN 2027", "TBA (2026: 10–12 Oct, Porto)", "TBA", "expected Jun 2027 (2026: 15 Jun)", "IEEE Xplore, 4 pp", "watch — 2024 Chicago, 2025 LA"),
    ("IEEE/ACM CHASE 2027", "TBA (2026: 4–6 Aug, Pittsburgh)", "US historically", "expected Feb 2027 (2026: 16 Feb)", "IEEE / ACM DL", "watch"),
    ("MLHC 2027", "TBA (2026: 12–14 Aug, Baltimore)", "US historically", "expected Apr 2027 (2026: 17 Apr)", "PMLR", "watch"),
    ("NEBEC 2027 (Northeast Bioeng. Conf.)", "Apr 2027, host TBA (2026: Temple)", "Northeast rotation", "expected Feb 2027 (2026: 20 Feb)", "abstracts in recent years", "watch — fails “papers” as run"),
    ("AMIA 2027 Amplify", "12–15 Apr 2027", "Atlanta", "3 Sep 2026 — open now", "archived proceedings", "US ✓ · New England ✗"),
    ("AMIA 2027 Annual Symposium", "6–10 Nov 2027", "San Diego", "expected Mar 2027", "archived proceedings", "US ✓ · New England ✗"),
    ("IEEE SPMB 2027", "Dec 2027 (2026: 5 Dec)", "Philadelphia (Temple)", "expected Jul 2027 (2026: 1 Jul)", "IEEE Xplore", "US ✓ · New England ✗"),
    ("IEEE EMBC 2027", "11–15 Jul 2027", "Singapore", "expected Jan–Feb 2027", "IEEE Xplore", "outside the US"),
    ("IEEE EMBS NER 2027", "7–10 Sep 2027", "Milan", "expected Jun 2027 (2025: 10 Jun)", "IEEE Xplore", "outside the US"),
    ("IEEE I2MTC 2027", "17–20 May 2027", "Chongqing", "expected Nov 2026", "IEEE Xplore", "outside the US"),
    ("IEEE MeMeA 2027", "2027, dates TBA", "Rome", "24 Jan 2027", "IEEE Xplore", "outside the US"),
    ("IEEE EMBS BHI 2026", "12–15 Dec 2026", "Hong Kong", "closed (3 Jul 2026)", "IEEE Xplore", "outside the US"),
    ("IEEE SENSORS 2026", "25–28 Oct 2026", "Rotterdam", "closed", "IEEE Xplore", "outside the US"),
    ("ML4H 2026 (with NeurIPS)", "6–7 Dec 2026", "Sydney", "~Sep 2026 (confirm)", "PMLR", "outside the US"),
]


# ----------------------------------------------------------------------------- notes
NOTES: dict[str, str] = {}

NOTES["title"] = """\
Good afternoon. This is a short update on where the bruxism paper stands and why it looks so different from the manuscript we sent to Scientific Reports.

The one-sentence version: we set out to sell an audio-plus-EMG classifier for bruxism-related activities, and along the way we found that two things nobody had measured were defective: the signal entering the model, and the microphone channel itself. The EMG survived. The paper now audits the whole pipeline, prices every stage, and tells readers how to do the same on their own data.

I will go through: where we were, the three audits, the two defects, with the microphone evidence in some detail, the new scope and results, the advice we give readers, what this means for the group and the NIH direction, venues with their dates and impact factors, and the decisions I need today. About twenty minutes, then discussion.

Everything I quote is from v8 of the manuscript (26 Aug), the advisor briefing (19 Aug) and the audit notes; nothing here is a new number."""

NOTES["where"] = """\
Where we were. The submitted manuscript, v1, was a detection story. It claimed 85 percent accuracy and an 80 percent F1 across four activities, movement, clenching, grinding and chewing, with no resting class, and it credited the microphone with a seventeen-point gain. It called itself real-time.

Scientific Reports rejected it on 3 July. Two reviewers were positive, two were not, and the editor sided with the two critics. Reviewer 2 is a bruxism clinician: the definition we used, bruxism equals tooth contact, is outdated; the phenotype was unspecified; and they wanted the STAB tool, the biopsychosocial model and TMD standard-of-care discussed. Reviewer 4 was technical and, frankly, correct: chewing is easy and inflates the headline; without a resting class it is not detection; the sentence about generalising to unseen individuals over-claims from five people; and the events were emulated.

Versions 2 and 3 were the resubmission draft. They fixed the protocol problems, added quiet rest, nested leave-one-subject-out, participant-level metrics, and wrote the clinical sections Reviewer 2 asked for. But the story was still fusion: the fused model at 82.1 percent accuracy, and RQ2, how much does audio add, and RQ3, does our dual-branch network beat the baselines, were still the spine.

What none of those versions had done was measure the signal or the channel the whole story rested on."""

NOTES["timeline"] = """\
How we got here, in six weeks and three audits.

First, the code audit at the end of July. When we rebuilt the prototype as a proper package, we found the held-out subject had been used for early stopping and checkpoint selection and then reported as the test score. There was window-level K-fold on overlapping windows, a focal-loss bug, mislabelled wavelet bands. And the published 85 percent confusion matrix cannot be reproduced under any labelling policy: its total equals the count including 595 rest windows that the matrix does not contain. The rebuild seals the outer fold structurally, has 196 tests, and regenerates every number from a saved prediction ledger.

Second, the signal audit on 3 August. The acquisition hardware had already notched 60 hertz. The interference that survived sat at 180, 300 and 420. Our textbook chain notched 60, the one frequency that was absent, and passed everything else. In the quiet-rest recordings, 91 to 99.8 percent of in-band power was mains.

Third, the channel audit on 12 August, which I will show in detail: the microphone column is 37 waveforms replayed across 100 recordings.

The versions after that, v4 to v8, are the paper being rewritten around what the data can actually support: the microphone came out in v5 on 15 August, and the advisor briefing on 19 August put that direction, with the evidence, to the advisor."""

NOTES["defect1"] = """\
Defect one, in one figure. Panel a is the superseded chain drawn over the measured spectrum of the recordings. The notch lands at 60 hertz, where the hardware had already removed everything, and the spikes at 180, 300 and 420 pass straight through. Panel b is the corrected chain, seven constant-width notches on the harmonics. Panel c is the residual mains share per participant: above 0.9 before, under 0.01 after.

What it was worth: the matched network went from 43.5 to 72.8 percent macro-F1 on byte-identical windows, folds and labels, 29.3 points. Quiet-rest windows misread as clenching or grinding fell from 30 percent to 6. And two participants who had been at or below chance, P1 at 18 and P2 at 5 percent macro-F1, are now in the 61 to 82 range with everyone else.

Two things to stress. The chain looked correct on its own response plot; the defect only appeared when the response was drawn over the data. And it did not look like a signal problem; it looked like between-participant heterogeneity, which is the most familiar failure mode in this literature and the one that motivates transfer learning and bigger cohorts. That is why the check is now a pipeline stage that refuses to run, not a review step."""

NOTES["mic_identity"] = """\
Defect two, starting with the whole finding in one picture. The figure compares every recording against every other by exact waveform identity. Left is the microphone, right is EMG channel 1 from the same 100 files. Red means two recordings contain the same samples and belong to different people. The microphone panel is striped with red across every participant boundary; the EMG panel has nothing off the diagonal.

The numbers: 37 distinct waveforms in 100 recordings; 83 recordings share a waveform with a different participant; of 63 pairs we tested, all 63 are exact circular rotations, the largest residual after alignment is zero. All four EMG channels are 100 out of 100 distinct, so every EMG result stands.

The consequence is that leave-one-subject-out never held out the audio. The next four slides show what that looks like in the signals themselves, what broke, what it did to the numbers, and why the channel would not have been usable audio even if it were unique.

The decision at the end of it: we dropped the microphone from every result, recorded RQ2 as untested rather than refuted, and kept the check, a rotation-invariant fingerprint of every channel on every manifest build that refuses a run if a held-out participant's signal is in its own training set."""

NOTES["mic_exhibit"] = """\
This is the strongest single exhibit, and it is the one to linger on if anyone doubts the finding.

Panel A is the microphone channel as stored, for the deviation-left-right condition, from five different people, recorded in five sessions across three days in August 2025. Look at the shapes: the same dropout, the same bumps, just displaced in time. They look shifted, not different.

Panel B undoes a circular shift for each trace, 1.90 seconds for S02, 3.12 for S03, minus 0.99 for S04, 1.90 for S05, and the five collapse onto one line.

Panel C is the difference from S01 after alignment. The largest absolute difference anywhere in the full 60-second recording is zero counts. Not correlated. Not similar. Identical, every sample, bit for bit.

Panel D is the EMG from those same five files: five clearly different people. Whatever failed, failed only on the microphone column.

One more detail worth saying out loud: S02 and S05 needed the same 1.90-second shift. That is what you would see if S02's column had been copied directly into S05's files, and the audit confirms it: S05's four duplicated recordings carry S02's exact sample offsets."""

NOTES["mic_offsets"] = """\
What broke, mechanically. We can say more than the files are duplicated, and the advisor will ask whether this could recur.

Left: the shifts across all 63 confirmed pairs. They are small, mostly within a few seconds, and they wrap in both directions around zero. A resampling or export bug does not produce circular rotations; a buffer whose read pointer is not reset between sessions does. That is a ring-buffer trace.

Right: how the copies are organised. Each row is one waveform stored under several participants, under the same condition. Four waveforms appear under all five people; sixteen more under four. The copies are grouped by condition, not by session, which is what turns a duplicated file into a labelled-data problem: the model sees the same audio under the same label in training and in test.

Corroboration from outside the CSVs: the three numpy companions we have, all from S01, match the CSV exactly on EMG and trigger and have an all-zero microphone column, so the acquisition array held no microphone data. And all 100 video files parse to one video stream and zero audio streams. There is nothing to recover the true audio from.

The requirements for the next collection follow directly: audio in its own file on its own clock, never a shared buffer, a hardware sync marker on both streams, and a fingerprint check at ingest."""

NOTES["mic_exposure"] = """\
What it did to the results. Left: for each held-out participant, how many of their 20 recordings carry a microphone waveform that was also in the training set. S01, S03 and S04: all twenty. S02: nineteen. S05: four. Leave-one-subject-out held out the participant; it did not hold out their audio. For the four S01-to-S04 rest recordings, which share a single waveform, every fold had already been trained on the exact rest audio three times.

Right: S05 is the control nobody designed. It is the only participant whose audio was mostly not copied, and it is the worst participant on audio-only macro-F1, 0.354, while being the best on EMG-only, 0.805. If S05 were simply a hard subject it would be bad on both. That dissociation is the signature of leakage.

The table gives the per-participant numbers from the ablation ledger. And the one cross-condition swap behaves as predicted: S05's protrusion recording carries the incisor-clench waveform, and its 63 audio-only predictions across three seeds were clenching, grinding and rest, never movement.

So the audio-only and fusion rows of the old Table 4 are withdrawn, along with the reading that the microphone helped separate quiet rest from tooth contact, and three to four screening points that came from seven microphone features. The direction of the bias is not even uniform, so we cannot say the true value is lower; we can only say the number is not what its name says."""

NOTES["mic_not_audio"] = """\
Could we still use it as audio if it were not duplicated? No, and this slide is why.

Left: where the microphone's energy is. Ninety-six percent of its power lies below 10 hertz, the shaded band. The EMG, for comparison, is broadband to 450. The production 20-hertz high-pass, the dashed line, throws away almost everything the channel contains: a median 1.19 percent of the variance survives.

Middle: all 100 recordings agree, 91 to 99 percent of power below 10 hertz in every one. A microphone recording of tooth contact puts its energy in the kilohertz range; at a 1200-hertz sampling rate the Nyquist limit is 600, so that range does not exist in these files. This is a sound-level envelope, not an acoustic waveform.

Right: the samples themselves. Integer-valued, one-count steps, 15 to 145 distinct values in a minute; a three-second excerpt here takes two values. After the high-pass, what survives for clenching and grinding sits at or below the one-count quantisation floor, so for the two classes the study exists to measure the audio branch was being fed converter dither."""

NOTES["mic_contrast"] = """\
This is the version that needs no signal-processing background, and the one I would show a clinician.

One point per recording: each activity's amplitude divided by that same participant's own quiet-rest session, on a log scale. Above the dashed line is louder than rest; below it is quieter.

Left, the EMG in the analysis band: every activity sits above that person's own quiet rest, and the ordering is the physiological one, chewing 5.8 times, grinding 4.1, clenching 2.4, movement 1.7.

Right, the raw microphone channel: chewing 1.8 times, movement 1.3, and then grinding at 0.44 and clenching at 0.30. Clenching and grinding, the two behaviours the study exists to measure, register quieter than a closed, silent mouth. No working microphone does that.

That contrast, side by side on identical recordings, is the fastest way to see that the column is not what it was labelled as, and it is the check we would run first on any new collection."""

NOTES["scope"] = """\
The new scope. The title is now "Where does the performance come from? A stage-by-stage analysis of a biosignal classification pipeline, with surface-EMG jaw-activity recognition as the case study."

The thesis is simple. Pipelines are reported through a summary accuracy and a block diagram, which says how well a pipeline scored and nothing about which stage earned it. We vary one stage at a time on identical windows, folds and labels, and put every stage on one scale.

Four research questions. RQ1: how much of what the classifier learns is physiology and how much is mains. RQ2: can four EMG channels separate quiet rest from four instructed jaw activities on an unseen participant. RQ3: which part of the model does the work, representation, architecture or parameter count. RQ4: how do those compare with choices fixed before training, window length, temporal context, normalisation scope.

On the right, what is kept, gone and new. Kept: the five-class EMG task with rest, the nested LOSO protocol, the compact wavelet CNN, and the dataset, which is now foregrounded with a section on why it had to be collected. Gone: fusion and the microphone, the detection and real-time claims, and the clinical sections we wrote for Reviewer 2, because the target venue is engineering. New: the filter contrast, the architecture sweep, the pre-model choices, the budget table, guards that refuse, a positioning table against ten prior studies, and a procedure section that transfers to EEG, ECG and inertial pipelines."""

NOTES["results"] = """\
What stands. The pipeline is ordinary on purpose: notch bank, band-pass, one-second windows z-scored with training-participant statistics, a db4 wavelet decomposition, three small convolutional branches on the A4, D3 and D1 bands, and an MLP. 5,901 parameters, about a millisecond per window on a CPU.

On participants it never saw, nested LOSO with three seeds: 81.7 percent accuracy, 72.0 macro-F1, 0.958 macro AUC, on 6,173 windows. Rest is the best-recalled class at 91 percent. Movement is the weak one, and it is a precision problem: it is the smallest class and absorbs chewing windows. Clenching and grinding are the hard pair, as they should be.

The number I would point the clinicians at: 7.7 percent of quiet-rest windows land on the tooth-contact side. Most work in this area classifies activities against each other and cannot say anything about false alarms; this is the closest analogue our design can produce.

The architecture sweep at the bottom answers Reviewer 4's question about baselines. Both wavelet models placed above both raw-signal models, and the largest model placed third. The extended model, which is our model with the time axis kept and a rhythm head, is the strongest at 75.0, but the 2.8-point edge rests on three of five participants, so it is reported as secondary and the compact model stays the characterised reference."""

NOTES["budget"] = """\
This is the main result of the paper: the budget. Six choices priced on the same 6,173 windows and five folds, in points of participant-level macro-F1.

Signal quality sets the ceiling at 29.3 points. Below that ceiling nothing helps: through the superseded chain the best model scored 43.5, which is lower than the 61.3 the weakest baseline reaches on the corrected signal. No architecture could recover what the filter passed through.

Above the ceiling, what a model can represent mattered more than how big it was: the learned wavelet representation beats the band energies it is built from by 8.1 points; an explicit rhythm representation adds 2.8; and 3.2 times the parameters on the raw waveform loses 3.3.

And two choices fixed before any model was trained, temporal context at 7.4 and normalisation scope at 9.1, are worth as much as the architecture or more. Neither is a modelling decision.

The caveats are in the paper and I will say them here too: the entries are not additive; three rows are screening estimates, optimistic in level; and the magnitudes belong to this cohort and this lab. What transfers is the ordering and the practice of measuring it."""

NOTES["advice"] = """\
This is the advice we give readers, and it is the part that generalises. Five steps, none specific to EMG or bruxism.

One: audit the signal against what the analysis assumes, on the recordings you have. Draw every filter response over the measured spectrum, not alone. Fingerprint every channel of every recording. Publish the interference statistic beside the accuracy.

Two: fix the data, meaning the windows, folds and labels, verify that they are identical ledger to ledger, and vary one stage at a time.

Three: price the choices made before training: window length, temporal context, normalisation scope, label source. Here two of them were worth more than the architecture, and nobody in this literature prices them.

Four: report a budget beside the accuracy, one row per stage, naming its harness and its baseline, and say what is not additive.

Five: keep a prediction ledger and regenerate every number and figure from it, so a figure cannot disagree with the table it depicts.

And build guards that refuse rather than warn: the leakage assertion on every fold, the check that no branch reads a band its own filter deleted, and the check that no run trains on a held-out participant's own waveform. The whole thing costs one pass over the data.

Both of our defects had this shape: a correct check, scoped one step too narrowly, to the filter rather than the data, to the file rather than the dataset."""

NOTES["group"] = """\
What this means for the group, and for the NIH direction, which I know is the real concern.

Why the reframe is right. A leave-one-subject-out number on a channel that was not held out is the specific kind of result that gets papers corrected. A caveat cannot fix it, because the number is mislabelled, not merely uncertain. The EMG-only paper has a clear question, a quantified answer, and a method others can adopt. And finding two measurement-chain defects in one dataset, with the checks that catch each, reads as a group that measures its own instruments. The bad version of this story is the one where someone else finds it.

Why it helps a microphone-plus-EMG grant. Nothing here refutes the acoustic hypothesis; these files contain no usable audio, so it is untested. It removes a latent leakage artifact before it could sit in a Preliminary Studies section. It is concrete rigor-and-reproducibility evidence: a measurement, a versioned policy, tests, an ingest gate, which reviewers score. And it converts "we will collect audio" into a protocol-plus-validation aim; the nine-requirement acquisition spec already exists in the repo.

The proposal: publish the EMG paper now; run a small audio re-collection pilot, five to ten people, against that spec, with audio at 16 kilohertz or better in its own file on its own clock, a hardware sync marker, and a fingerprint check at ingest; and let that pilot be the grant's preliminary data. It gives a better audio aim than the contaminated data ever could, for the cost of one pilot."""

NOTES["journals"] = """\
Where to submit, starting with journals, because the recommendation is a journal first: rolling submission, no location constraint, and a methods-and-audit paper suits twenty pages better than six. The impact factors are the 2025 Journal Citation Reports values released in June, cross-checked against the publishers' own pages where those show them; fees are the 2026 open-access charges.

First choice is JBHI, impact factor 7.7: it publishes exactly this kind of sensor-to-decision pipeline work and values the leakage-free protocol. TBME, 4.4, if we want the strongest line on a CV and can wait. Physiological Measurement, 2.5, is a genuinely good fit for the audit framing; its scope statement is literally the development and validation of methods of measurement, and it states a median 49 days to first decision. If we need something citable before the grant is written, OJEMB at 3.1 and IEEE Access at 4.2 are the fast, open-access options at 2,160 dollars each for 2026, less with IEEE or EMBS membership. Computers in Biology and Medicine, 6.3, Biomedical Signal Processing and Control, 5.7, and the Journal of Electromyography and Kinesiology, 2.7, are the methods-and-signal alternatives; JEK is where the notch finding would land hardest with EMG people.

Below the table: Computer Methods and Programs in Biomedicine at 6.4, MBEC at 3.1, IEEE Sensors Journal at 4.5, and the Journal of Oral Rehabilitation at 3.8, which remains the strategic option if we want the work visible to the clinical community, but would need reframing."""

NOTES["conferences"] = """\
Conferences, with dates and deadlines. I applied three conditions: in the US, in New England, and full papers published in proceedings rather than abstracts. As of the check date, nothing announced for 2026 or 2027 meets all three.

The top block is the watch-list. BHI 2027 is the best fit: confirmed for the US in November 2027, city to be announced, full papers in Xplore, and on this year's pattern the deadline will fall around June 2027. BSN 2027 has not been announced; it was Chicago in 2024 and Los Angeles in 2025 before Porto, so a US edition is plausible. CHASE and MLHC are US-based with deadlines around February and April. NEBEC rotates through the Northeast, but it has run on abstracts in recent years, so it fails the full-paper condition unless the 2027 host changes the format.

The middle block is US but not New England. AMIA Amplify in Atlanta is the only deadline open right now, 3 September, and the fit is moderate. SPMB in Philadelphia closed on 1 July.

The bottom block is everything else we checked, all outside the US: EMBC in Singapore, NER in Milan, I2MTC in Chongqing, MeMeA in Rome with a January deadline, BHI 2026 in Hong Kong, SENSORS in Rotterdam, ML4H in Sydney.

The practical reading: submit to a journal this fall, and hold BHI 2027 as the conference target once its call appears."""

NOTES["decisions"] = """\
Four decisions I need from this meeting.

One: approve the v8 framing and title, the general stage-by-stage analysis with EMG jaw activity as the case study. It is 26 pages with references from page 20, and every number traces to a saved artifact.

Two: pick the journal. My recommendation is JBHI; OJEMB or IEEE Access if being citable before the grant matters more than the venue.

Three: approve the audio re-collection pilot and its acceptance criteria. The grant's audio aim rests on it, and the spec is already written.

Four: close the open items before submission. The hardware, gain and ADC documentation, which the paper currently reports as undocumented; the IRB identifier, where the records carry conflicting values; the data-release scope; and co-author sign-off on the dropped microphone and clinical sections.

Nothing in the EMG results changes if the answers are yes. Everything in the microphone story changes if the pilot is not run. Thank you; questions."""

SLIDE_TITLES = {
    "title": "Title — Where Does the Performance Come From?",
    "where": "Where we were: v1–v3 sold a fusion classifier for bruxism",
    "timeline": "How we got here: three audits, six weeks, one surviving modality",
    "defect1": "Defect 1: the 60-Hz notch removed the one frequency that was already gone",
    "mic_identity": "Defect 2: the microphone column is not per-participant audio",
    "mic_exhibit": "Microphone: one waveform, replayed under five participants",
    "mic_offsets": "Microphone: what broke — ring-buffer offsets, copies grouped by condition",
    "mic_exposure": "Microphone: the audio ablation scored the training set",
    "mic_not_audio": "Microphone: not usable audio even if it were unique",
    "mic_contrast": "Microphone: clenching registers quieter than silence",
    "scope": "The new scope (v8)",
    "results": "What stands: EMG alone, on participants the model never saw",
    "budget": "The main result: signal quality sets the ceiling; representation beats size",
    "advice": "What we tell readers: the five-step procedure",
    "group": "What it means for the group and the NIH direction",
    "journals": "Where to submit: journals, with impact factors and fees",
    "conferences": "Conference calendar: dates, deadlines, and the New England test",
    "decisions": "Decisions",
}
TIMING = {"title": "0:45", "where": "1:45", "timeline": "1:30", "defect1": "1:30", "mic_identity": "1:30",
          "mic_exhibit": "1:30", "mic_offsets": "1:15", "mic_exposure": "1:30", "mic_not_audio": "1:15",
          "mic_contrast": "0:45", "scope": "1:30", "results": "1:45", "budget": "1:30", "advice": "1:30",
          "group": "1:45", "journals": "1:30", "conferences": "1:15", "decisions": "0:45"}


# ----------------------------------------------------------------------------- build
def build():
    ASSETS.mkdir(exist_ok=True)
    figs = {
        "filter": (PAPER_FIG / "filter_defect_and_correction.png", ASSETS / "filter_defect_and_correction.png"),
        "pipeline": (PAPER_FIG / "Pipeline_V2.png", ASSETS / "pipeline_v2.png"),
        "identity": (ADVISOR_FIG / "fig1_identity_matrix.png", ASSETS / "mic_identity_matrix.png"),
        "exhibit": (ADVISOR_FIG / "fig2_smoking_gun.png", ASSETS / "mic_one_waveform_five_participants.png"),
        "offsets": (ADVISOR_FIG / "fig7_offsets.png", ASSETS / "mic_offsets_and_groups.png"),
        "exposure": (ADVISOR_FIG / "fig4_loso_exposure.png", ASSETS / "mic_loso_exposure.png"),
        "not_audio": (ADVISOR_FIG / "fig5_not_audio.png", ASSETS / "mic_not_audio.png"),
        "contrast": (ADVISOR_FIG / "fig6_contrast.png", ASSETS / "mic_contrast_vs_rest.png"),
    }
    for src, dst in figs.values():
        if src.exists():
            shutil.copyfile(src, dst)
        elif not dst.exists():
            raise FileNotFoundError(src)
    budget_png = ASSETS / "budget_chart.png"
    make_budget_chart(budget_png)

    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    deck = Deck(prs, total=len(SLIDE_TITLES))

    # ---------------------------------------------------------------- title
    s = deck.title_slide("title", NOTES["title"])
    add_rect(s, Inches(0.6), Inches(1.9), Inches(0.09), Inches(3.4), fill=ACCENT)
    add_text(s, Inches(0.95), Inches(1.75), Inches(11.5), Inches(0.4),
             [dict(text=f"PROJECT UPDATE · {DATE.upper()}", size=12, bold=True, color=ACCENT)])
    add_text(s, Inches(0.95), Inches(2.15), Inches(11.5), Inches(1.4),
             [dict(text="Where Does the Performance Come From?", size=40, bold=True, color=INK)],
             line_spacing=1.0)
    add_text(s, Inches(0.95), Inches(3.35), Inches(11.2), Inches(1.3),
             [dict(text="The new scope of the bruxism paper (v8): from selling an audio–EMG classifier "
                        "to auditing the whole pipeline — what changed since the Scientific Reports "
                        "submission, why, and where it goes next", size=18, color=SLATE)],
             line_spacing=1.1)
    add_text(s, Inches(0.95), Inches(4.75), Inches(11.2), Inches(0.5),
             [dict(text="M. H. (Kye) Farhadi · Electrical, Computer & Biomedical Engineering, "
                        "University of Rhode Island", size=13, color=INK)])
    add_text(s, Inches(0.95), Inches(6.55), Inches(11.5), Inches(0.5),
             [dict(text="Sources: main_v8.tex (26 Aug) · advisor briefing (19 Aug) · audit notes "
                        "cause.md and audio.md · Scientific Reports decision (3 Jul) · venue facts "
                        f"checked {CHECKED}. No number here is new.", size=10, color=MUTED)])

    # ---------------------------------------------------------------- where we were
    s = deck.slide("where", "Where we were", "v1–v3 sold a fusion classifier for bruxism", NOTES["where"])
    colw = Inches(5.95)
    x2 = LM + colw + Inches(0.25)
    header_line(s, LM, Inches(1.6), colw, "v1 — submitted to Scientific Reports")
    add_text(s, LM, Inches(1.95), colw, Inches(2.6), [
        dict(runs=[("“Sensor-based Bruxism Detection with Dual-Branch Wavelet CNNs and Audio–EMG "
                    "Data Fusion”", {"italic": True})], bullet=True),
        dict(text="Claimed 85.0% accuracy / 80.3% F1 on four classes (movement, clench, grind, chew) "
                  "— no resting class", bullet=True),
        dict(text="Audio fusion credited with +17.4 points; “real-time”, ~15,000 parameters, "
                  "12 ms per window", bullet=True),
        dict(text="Framed as clinical screening: “bruxism detection”", bullet=True),
    ], size=12.5)
    header_line(s, LM, Inches(4.0), colw, "Rejected 3 Jul 2026 — 2 of 4 reviewers critical", color=BAD)
    add_text(s, LM, Inches(4.35), colw, Inches(2.4), [
        dict(runs=[("R2 (clinician): ", {"bold": True}),
                   ("tooth-contact-only definition is outdated; phenotype unspecified; discuss STAB, "
                    "the biopsychosocial model and TMD standard of care", {})], bullet=True),
        dict(runs=[("R4 (technical): ", {"bold": True}),
                   ("chewing inflates the headline; no resting class, so it is not detection; "
                    "“generalises to unseen individuals” over-claims N = 5; events emulated, "
                    "not natural", {})], bullet=True),
    ], size=12.5)
    header_line(s, x2, Inches(1.6), colw, "v2–v3 — the resubmission draft (Jul–Aug 2026)")
    add_text(s, x2, Inches(1.95), colw, Inches(3.6), [
        dict(text="Added quiet rest → five classes; nested leave-one-subject-out; participant-level "
                  "metrics; clinical terminology sections written for R2", bullet=True),
        dict(text="Still the same story: audio + EMG fusion, 82.1% accuracy / 73.6% macro-F1 / "
                  "0.962 AUC for the fused model", bullet=True),
        dict(text="RQ2 “how much does audio add?” and RQ3 “does the dual-branch CNN beat the "
                  "baselines?” remained the spine", bullet=True),
        dict(text="Every number came from a filter chain and a microphone channel nobody had measured",
             bullet=True),
    ], size=12.5)
    add_rect(s, LM, Inches(6.0), CW, Inches(0.85), fill=WASH, rounded=True, radius=0.1)
    add_text(s, LM + Inches(0.2), Inches(6.0), CW - Inches(0.4), Inches(0.85), [
        dict(text="The pitch rested on two things nobody had measured: the signal entering the model and "
                  "the microphone channel. Both turned out to be defective. The EMG survived.",
             size=13.5, color=INK, bold=True)], anchor=MSO_ANCHOR.MIDDLE)

    # ---------------------------------------------------------------- timeline
    s = deck.slide("timeline", "How we got here", "Three audits, six weeks, one surviving modality", NOTES["timeline"])
    events = [
        ("3 Jul", "Scientific Reports rejects v1"),
        ("27–28 Jul", "Prototype rebuilt as a package"),
        ("3 Aug", "Mains-harmonic defect found"),
        ("7–10 Aug", "Confirmatory runs, corrected chain"),
        ("12 Aug", "Microphone audit"),
        ("15–17 Aug", "v4–v7: mic dropped, pipeline framing"),
        ("19 Aug", "Advisor briefing"),
        ("25–26 Aug", "v8: general framing, 20 pp"),
    ]
    ty = Inches(2.05)
    add_rect(s, LM + Inches(0.3), ty, CW - Inches(0.6), Inches(0.03), fill=RULE)
    step = (CW - Inches(0.6)) / (len(events) - 1)
    for i, (d, e) in enumerate(events):
        cx = LM + Inches(0.3) + int(step * i)
        dot = s.shapes.add_shape(MSO_SHAPE.OVAL, cx - Inches(0.08), ty - Inches(0.065), Inches(0.16), Inches(0.16))
        dot.fill.solid()
        dot.fill.fore_color.rgb = ACCENT if i in (1, 2, 4) else SLATE
        dot.line.fill.background()
        dot.shadow.inherit = False
        add_text(s, cx - Inches(0.75), Inches(1.6), Inches(1.5), Inches(0.35),
                 [dict(text=d, size=10.5, bold=True, color=INK, align=PP_ALIGN.CENTER)], space_after=0)
        add_text(s, cx - Inches(0.78), Inches(2.2), Inches(1.56), Inches(0.6),
                 [dict(text=e, size=9.5, color=SLATE, align=PP_ALIGN.CENTER)], space_after=0, line_spacing=1.0)
    cards = [
        ("Code audit · 27–28 Jul", [
            "Prototype used the held-out subject for early stopping and checkpoint selection, then reported it as the test score",
            "Window-level K-fold on 50%-overlapping windows; focal-loss and wavelet-band bugs",
            "The published 85% matrix is irreproducible: its total counts 595 rest windows it does not contain",
            "Rebuilt: sealed outer folds, 196 tests, every number regenerated from a saved prediction ledger",
        ]),
        ("Signal audit · 3 Aug", [
            "Acquisition hardware had already notched 60 Hz; interference sat at 180 / 300 / 420 Hz",
            "The textbook chain notched 60 Hz — the one frequency that was absent — and passed the rest",
            "91–99.8% of in-band quiet-rest power was mains; two participants were unclassifiable",
            "Fix: constant-width notch bank at every harmonic, selected on an interference criterion, not on score",
        ]),
        ("Channel audit · 12 Aug", [
            "Fingerprint of every channel of every recording: EMG 100/100 distinct, microphone 37/100",
            "83 recordings carry another participant’s audio, rotated — LOSO never held out the audio",
            "Cause: buffer reuse in the acquisition path; no second copy exists (.avi has no audio stream)",
            "Outcome: microphone dropped from the paper in v5 (15 Aug), put to the advisor 19 Aug; the acoustic hypothesis is untested, not refuted",
        ]),
    ]
    cw_in = 3.9
    for i, (title, bullets) in enumerate(cards):
        x = LM + Inches((cw_in + 0.215) * i)
        add_rect(s, x, Inches(3.05), Inches(cw_in), Inches(3.75), fill=WASH, rounded=True, radius=0.05)
        add_text(s, x + Inches(0.15), Inches(3.12), Inches(cw_in - 0.3), Inches(0.35),
                 [dict(text=title, size=13, bold=True, color=ACCENT)], space_after=0)
        add_text(s, x + Inches(0.15), Inches(3.5), Inches(cw_in - 0.3), Inches(3.25),
                 [dict(text=b, bullet=True) for b in bullets], size=11, space_after=5)

    # ---------------------------------------------------------------- defect 1
    s = deck.slide("defect1", "Defect 1 — the EMG filter chain",
                   "The 60-Hz notch removed the one frequency that was already gone", NOTES["defect1"])
    add_image(s, figs["filter"][1], LM, Inches(1.55), w=Inches(9.05))
    tx = LM + Inches(9.3)
    tw = CW - Inches(9.3)
    stat_tile(s, tx, Inches(1.55), tw, Inches(1.02), "+29.3 pts",
              "macro-F1, 43.5% → 72.8%, same windows, folds and labels",
              value_color=GOOD, vsize=24, lsize=10.5)
    stat_tile(s, tx, Inches(2.62), tw, Inches(1.02), "30.2% → 6.1%",
              "quiet-rest windows misread as clenching or grinding",
              value_color=GOOD, vsize=24, lsize=10.5)
    stat_tile(s, tx, Inches(3.69), tw, Inches(1.02), "2 of 5 → 0 of 5",
              "participants at or below chance before (P1 18%, P2 5% macro-F1)",
              value_color=GOOD, vsize=22, lsize=10.5)
    add_text(s, LM, Inches(4.85), CW, Inches(2.1), [
        dict(text="The chain looked textbook-correct on its own response plot; the defect appeared only when "
                  "the response was drawn over the measured spectrum of the recordings", bullet=True),
        dict(text="It presented as between-participant heterogeneity — the most familiar failure mode in "
                  "this literature — not as a signal problem", bullet=True),
        dict(text="Three estimators of very different capacity moved 22–29 points on the same windows: a "
                  "ceiling, not a model effect", bullet=True),
        dict(text="Now a pipeline stage: the residual-mains statistic is published beside every accuracy, and "
                  "a band-inside-passband guard refuses to run", bullet=True),
    ], size=12.5, space_after=5)

    # ---------------------------------------------------------------- mic 1: identity matrix
    s = deck.slide("mic_identity", "Defect 2 — the microphone channel (1 of 6)",
                   "The microphone column is not per-participant audio", NOTES["mic_identity"])
    add_image(s, figs["identity"][1], LM, Inches(1.55), w=Inches(6.7))
    rx = LM + Inches(6.95)
    rw = CW - Inches(6.95)
    header_line(s, rx, Inches(1.5), rw, "Measured from the raw CSVs (recomputed 19 Aug)")
    add_text(s, rx, Inches(1.85), rw, Inches(3.2), [
        dict(text="37 distinct waveforms in 100 recordings; 83 recordings share a waveform with a different "
                  "participant; 63 of 63 tested pairs are exact circular rotations (residual 0)", bullet=True),
        dict(text="All four EMG channels: 100 of 100 distinct, zero sharing — every EMG result stands",
             bullet=True),
        dict(text="LOSO never held out the audio: the old fusion and audio-only numbers scored training data",
             bullet=True),
        dict(text="Not audio even if unique: 96% of power below 10 Hz, 1-count quantisation, clenching "
                  "quieter than silent rest (0.30×)", bullet=True),
        dict(text="Not repairable: the .avi files carry no audio stream; the .npy companions have an "
                  "all-zero mic column", bullet=True),
    ], size=11.5, space_after=4)
    header_line(s, rx, Inches(5.05), rw, "What we did", color=GOOD)
    add_text(s, rx, Inches(5.4), rw, Inches(1.5), [
        dict(text="Dropped the microphone from every result; RQ2 recorded as untested, not refuted", bullet=True),
        dict(text="Kept the check: a rotation-invariant fingerprint of every channel on every manifest build; "
                  "a run is refused if a held-out participant’s signal is in its own training set", bullet=True),
    ], size=11.5, space_after=4)
    caption(s, LM, Inches(5.95), Inches(6.7),
            "Every recording against every other, by exact waveform identity. Red: identical samples under "
            "two different participants. Left, the microphone; right, EMG channel 1 from the same files.")

    # ---------------------------------------------------------------- mic 2: the exhibit
    s = deck.slide("mic_exhibit", "Defect 2 — the microphone channel (2 of 6)",
                   "One microphone waveform, replayed under five different participants", NOTES["mic_exhibit"])
    add_image(s, figs["exhibit"][1], LM, Inches(1.5), h=Inches(5.4))
    rx = LM + Inches(6.15)
    rw = CW - Inches(6.15)
    add_text(s, rx, Inches(1.55), rw, Inches(5.3), [
        dict(runs=[("A · As stored. ", {"bold": True}),
                   ("The deviation-left-right condition from five people, five sessions, three days "
                    "(4–7 Aug 2025). The traces look shifted in time, not different.", {})], space_after=8),
        dict(runs=[("B · After undoing a circular shift ", {"bold": True}),
                   ("(S02 +1.90 s, S03 +3.12 s, S04 −0.99 s, S05 +1.90 s) the five traces coincide exactly.", {})],
             space_after=8),
        dict(runs=[("C · Difference from S01 after alignment: 0 counts ", {"bold": True}),
                   ("anywhere in the full 60-s recording. Not correlated, not similar — identical, bit for bit.", {})],
             space_after=8),
        dict(runs=[("D · The EMG from the same five files ", {"bold": True}),
                   ("is five clearly different people. Whatever failed, failed only on the microphone column.", {})],
             space_after=8),
        dict(runs=[("Same +1.90 s for S02 and S05: ", {"bold": True}),
                   ("what a direct copy of S02’s column into S05’s files would produce. The audit confirms it — "
                    "S05’s four duplicated recordings carry S02’s exact sample offsets (5,586 / 69,009 / "
                    "2,279 / 69,728).", {})], space_after=0),
    ], size=12, line_spacing=1.08)

    # ---------------------------------------------------------------- mic 3: offsets / cause
    s = deck.slide("mic_offsets", "Defect 2 — the microphone channel (3 of 6)",
                   "What broke: ring-buffer offsets, copies grouped by condition", NOTES["mic_offsets"])
    add_image(s, figs["offsets"][1], LM, Inches(1.5), w=CW)
    y0 = Inches(5.85)
    cw_in = (12.133 - 0.3) / 3
    cols = [
        ("Exact rotations, small offsets",
         "All 63 tested pairs match bit for bit once shifted. Shifts run −7.5 to +10 s and wrap both ways "
         "around zero: a ring buffer whose read pointer was never reset between sessions, not a resampling "
         "or export step."),
        ("Grouped by condition, not session",
         "Four waveforms sit under all five participants and sixteen under four, always under the same "
         "condition — which makes it a labelled-data problem, not a repeated file."),
        ("Nothing to recover from",
         "The three .npy companions (S01) match the CSV on EMG and trigger and hold an all-zero mic column; "
         "all 100 .avi files carry zero audio streams."),
    ]
    for i, (h, b) in enumerate(cols):
        x = LM + Inches((cw_in + 0.15) * i)
        add_text(s, x, y0, Inches(cw_in), Inches(0.3), [dict(text=h, size=11.5, bold=True, color=ACCENT)], space_after=0)
        add_text(s, x, y0 + Inches(0.3), Inches(cw_in), Inches(0.9), [dict(text=b, size=10.5, color=INK)],
                 line_spacing=1.03, space_after=0)

    # ---------------------------------------------------------------- mic 4: LOSO exposure
    s = deck.slide("mic_exposure", "Defect 2 — the microphone channel (4 of 6)",
                   "The audio ablation scored the training set", NOTES["mic_exposure"])
    add_image(s, figs["exposure"][1], LM, Inches(1.5), w=Inches(8.1))
    tx = LM + Inches(8.35)
    add_table(s, tx, Inches(1.55), [0.95, 0.95, 0.95, 0.95], [
        ["Held out", "Mic dup.", "Audio F1", "EMG F1"],
        ["S01", "20 / 20", "0.423", "0.632"],
        ["S02", "19 / 20", "0.425", "0.726"],
        ["S03", "20 / 20", "0.522", "0.713"],
        ["S04", "20 / 20", "0.414", "0.722"],
        ["S05", "4 / 20", "0.354", "0.805"],
    ], size=10, row_h=0.27, col_align=[None, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER], bold_first_col=True)
    caption(s, tx, Inches(3.25), Inches(3.8),
            "Ablation ledger modality_and_no_chewing_…cead62e4, five-class task, three-seed means "
            "(audio.md §1.6). Dup.: recordings whose mic waveform also appears under another participant.")
    add_text(s, LM, Inches(4.95), CW, Inches(2.0), [
        dict(text="LOSO held out the participant, not their audio: for S01–S04 essentially every held-out audio "
                  "window was also in training under the same label; the four S01–S04 rest recordings share one "
                  "waveform", bullet=True),
        dict(text="S05 — the only participant whose audio was mostly not copied — is the worst on audio and the "
                  "best on EMG: the dissociation leakage predicts, not a hard subject", bullet=True),
        dict(text="The one cross-condition swap behaves as predicted: S05’s protrusion recording carries the "
                  "incisor-clench waveform; its 63 audio-only predictions were clench, grind or rest — 0 movement",
             bullet=True),
        dict(text="Withdrawn: the audio-only and fusion rows of the old Table 4, the “audio helps rest” reading, "
                  "and 3–4 screening points from seven microphone features. The bias is not uniform in direction, "
                  "so no corrected number can be stated", bullet=True),
    ], size=11.5, space_after=4)

    # ---------------------------------------------------------------- mic 5: not audio
    s = deck.slide("mic_not_audio", "Defect 2 — the microphone channel (5 of 6)",
                   "Even if it were not duplicated, this channel is not usable audio", NOTES["mic_not_audio"])
    add_image(s, figs["not_audio"][1], LM, Inches(1.5), w=CW)
    y0 = Inches(5.75)
    cols = [
        ("Where the energy is",
         "96% of the microphone’s power (median; 91–99% across all 100 recordings) lies below 10 Hz. Tooth "
         "contact radiates in the kilohertz range; at 1200 Hz the Nyquist limit is 600 Hz, so that band does "
         "not exist in these files."),
        ("What the pipeline kept",
         "The production 20-Hz high-pass retains a median 1.19% of the channel’s variance — it removes the only "
         "band that carried condition information and keeps the band that does not."),
        ("What the samples are",
         "Integer-valued with a 1-count step; 15–145 distinct values per minute. After the high-pass, clenching "
         "and grinding sit at or below the quantisation floor: the audio branch was reading converter dither."),
    ]
    for i, (h, b) in enumerate(cols):
        x = LM + Inches((cw_in + 0.15) * i)
        add_text(s, x, y0, Inches(cw_in), Inches(0.3), [dict(text=h, size=11.5, bold=True, color=ACCENT)], space_after=0)
        add_text(s, x, y0 + Inches(0.3), Inches(cw_in), Inches(1.0), [dict(text=b, size=10.5, color=INK)],
                 line_spacing=1.03, space_after=0)

    # ---------------------------------------------------------------- mic 6: contrast
    s = deck.slide("mic_contrast", "Defect 2 — the microphone channel (6 of 6)",
                   "Every activity should be louder than silence; on the microphone, two are not",
                   NOTES["mic_contrast"])
    add_image(s, figs["contrast"][1], LM, Inches(1.5), w=Inches(8.6))
    rx = LM + Inches(8.85)
    rw = CW - Inches(8.85)
    add_text(s, rx, Inches(1.55), rw, Inches(5.3), [
        dict(runs=[("One point per recording: ", {"bold": True}),
                   ("each activity’s amplitude divided by that same participant’s own quiet-rest session. "
                    "Above the line is louder than rest.", {})], space_after=9),
        dict(runs=[("EMG (left): ", {"bold": True}),
                   ("every activity sits above that person’s rest, in the physiological order — chewing 5.8×, "
                    "grinding 4.1×, clenching 2.4×, movement 1.7×.", {})], space_after=9),
        dict(runs=[("Microphone (right): ", {"bold": True}),
                   ("chewing 1.8×, movement 1.3× — then grinding 0.44× and clenching 0.30×. The two behaviours "
                    "the study exists to measure register quieter than a closed, silent mouth. No working "
                    "microphone does that.", {})], space_after=9),
        dict(runs=[("Why it matters: ", {"bold": True}),
                   ("this needs no signal-processing background, uses identical recordings for both channels, "
                    "and is the first check to run on any new collection.", {})], space_after=0),
    ], size=12, line_spacing=1.08)

    # ---------------------------------------------------------------- new scope
    s = deck.slide("scope", "The new scope (v8)",
                   "From “does our classifier work?” to “where does the performance come from?”",
                   NOTES["scope"])
    lw = Inches(6.25)
    add_text(s, LM, Inches(1.55), lw, Inches(1.0), [
        dict(text="Where Does the Performance Come From? A Stage-by-Stage Analysis of a Biosignal Classification "
                  "Pipeline, with Surface-EMG Jaw-Activity Recognition as the Case Study",
             size=12.5, italic=True, color=SLATE)], line_spacing=1.05)
    add_text(s, LM, Inches(2.55), lw, Inches(1.15), [
        dict(text="Pipelines are reported through a summary accuracy and a block diagram. We vary one stage at a "
                  "time on identical windows, folds and labels, and price every stage on one scale.",
             size=14, bold=True, color=INK)], line_spacing=1.08)
    rqs = [
        ("RQ1", "How much of what the classifier learns is physiology, and how much is mains interference?"),
        ("RQ2", "Can four EMG channels separate quiet rest from four instructed jaw activities on an unseen participant?"),
        ("RQ3", "Which part of the model does the work: representation, architecture, or parameter count?"),
        ("RQ4", "How do those compare with choices fixed before training: window length, temporal context, normalisation scope?"),
    ]
    add_text(s, LM, Inches(3.75), lw, Inches(3.1),
             [dict(runs=[(f"{k}  ", {"bold": True, "color": ACCENT}), (v, {})], space_after=7) for k, v in rqs],
             size=12.5, line_spacing=1.05)
    bx = LM + lw + Inches(0.3)
    bw = CW - lw - Inches(0.3)
    boxes = [
        ("KEPT", GOOD, WASH_GOOD, [
            "Five-class EMG task with quiet rest · nested LOSO, 3 seeds, participant-level metrics",
            "Compact wavelet CNN (5,901 parameters) as the characterised reference",
            "The dataset and protocol — now foregrounded, with why they had to be collected",
        ]),
        ("GONE", BAD, WASH_BAD, [
            "Audio–EMG fusion and the microphone · “bruxism detection” and real-time claims",
            "RQ2-as-fusion and RQ3-as-“our model wins”",
            "The clinical / STAB / TMD sections written for Reviewer 2 (engineering venue)",
        ]),
        ("NEW", ACCENT, WASH, [
            "RQ1 filter contrast on identical windows · architecture sweep: representation vs. size",
            "RQ4 pre-model choices · the budget table (main result) · guards that refuse to run",
            "Positioning table (10 prior studies) · why-own-data section · a procedure that transfers to EEG / ECG / IMU",
        ]),
    ]
    by = Inches(1.55)
    bh = Inches(1.68)
    for title, col, fill, bullets in boxes:
        add_rect(s, bx, by, bw, bh, fill=fill, rounded=True, radius=0.06)
        add_rect(s, bx, by, Inches(0.09), bh, fill=col)
        add_text(s, bx + Inches(0.2), by + Inches(0.06), bw - Inches(0.3), Inches(0.3),
                 [dict(text=title, size=11, bold=True, color=col)], space_after=0)
        add_text(s, bx + Inches(0.2), by + Inches(0.36), bw - Inches(0.3), bh - Inches(0.4),
                 [dict(text=b, bullet=True) for b in bullets], size=11, space_after=3)
        by += bh + Inches(0.1)

    # ---------------------------------------------------------------- results
    s = deck.slide("results", "What stands", "EMG alone, on participants the model never saw", NOTES["results"])
    add_image(s, figs["pipeline"][1], LM, Inches(1.55), w=Inches(5.95))
    add_text(s, LM, Inches(4.86), Inches(5.95), Inches(0.5), [
        dict(text="Notch bank → 20–450 Hz band-pass → 1-s windows z-scored on training participants "
                  "→ db4 wavelet (A4, D3, D1) → three small CNN branches → MLP. "
                  "5,901 parameters, ~1 ms per window on a CPU.", size=10, color=SLATE)],
        line_spacing=1.02, space_after=0)
    rx = LM + Inches(6.25)
    rw = CW - Inches(6.25)
    tw = (rw - Inches(0.2)) / 3
    for i, (v, lab) in enumerate((("81.7%", "accuracy"), ("72.0%", "macro-F1"), ("0.958", "macro AUC"))):
        stat_tile(s, rx + (tw + Inches(0.1)) * i, Inches(1.55), tw, Inches(1.0), v, lab, vsize=24, lsize=11)
    add_text(s, rx, Inches(2.58), rw, Inches(0.3), [
        dict(text="Nested LOSO · 5 held-out participants × 3 seeds · 6,173 windows · "
                  "participant-level means", size=10, color=MUTED)], space_after=0)
    add_table(s, rx, Inches(2.92), [2.3, 1.05, 1.05, 1.45], [
        ["Class (windows)", "F1", "Recall", "AUC"],
        ["Rest (590)", "83.6%", "91.4%", "0.958"],
        ["Movement (333)", "46.0%", "69.8%", "0.937"],
        ["Clenching (799)", "68.1%", "66.0%", "0.936"],
        ["Grinding (816)", "68.6%", "69.5%", "0.928"],
        ["Chewing (3,635)", "91.1%", "85.8%", "0.981"],
    ], size=10, row_h=0.245, col_align=[None, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
    add_text(s, rx, Inches(4.45), rw, Inches(0.5), [
        dict(runs=[("7.7% of quiet-rest windows ", {"bold": True}),
                   ("land on the tooth-contact side — the closest thing to a false-alarm rate this design allows.", {})])],
        size=11, line_spacing=1.03, space_after=0)
    add_table(s, LM, Inches(5.4), [4.6, 1.3, 1.6, 1.3, 1.4], [
        ["Architecture sweep — identical inputs, 3 seeds", "Params", "Macro-F1", "AUC", "Forward"],
        ["Wavelet CNN, extended (time axis kept + rhythm head)", "17,981", "75.0 ± 1.3", "0.974", "2.21 ms"],
        ["Wavelet CNN, compact — the reference model", "5,901", "72.2 ± 1.5", "0.954", "1.02 ms"],
        ["Early-fusion CNN on the raw signals", "19,029", "68.9 ± 1.5", "0.949", "0.20 ms"],
        ["Bidirectional LSTM", "10,053", "60.2 ± 2.6", "0.898", "1.16 ms"],
    ], size=10, row_h=0.245, col_align=[None, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
    add_text(s, LM + Inches(10.4), Inches(5.4), CW - Inches(10.4), Inches(1.3), [
        dict(text="Both wavelet models above both raw-signal models; the largest model placed third. The extended "
                  "model is the reference with its time axis kept — reported as secondary (2.8 points, on "
                  "three of five participants).", size=9.5, color=SLATE)], line_spacing=1.03)

    # ---------------------------------------------------------------- budget
    s = deck.slide("budget", "The main result — Table 8 of v8, the budget",
                   "Signal quality sets the ceiling; representation beats size", NOTES["budget"])
    add_image(s, budget_png, LM, Inches(1.5), w=Inches(8.3))
    rx = LM + Inches(8.55)
    rw = CW - Inches(8.55)
    add_text(s, rx, Inches(1.55), rw, Inches(5.3), [
        dict(runs=[("Below the ceiling nothing helps. ", {"bold": True}),
                   ("Through the superseded chain the best model scored 43.5% — under the 61.3% the weakest "
                    "baseline reaches on the corrected signal.", {})], space_after=9),
        dict(runs=[("Representation beat size. ", {"bold": True}),
                   ("The two wavelet models placed first and second; 3.2× the parameters on the raw waveform "
                    "lost 3.3 points.", {})], space_after=9),
        dict(runs=[("Two choices fixed before training ", {"bold": True}),
                   ("— temporal context (+7.4) and normalisation scope (+9.1) — are worth as much as the "
                    "architecture (+8.1) or more. Neither is a modelling decision.", {})], space_after=9),
        dict(runs=[("Caveats, stated in the paper: ", {"bold": True}),
                   ("entries are not additive; three rows are screening estimates, optimistic in level; the "
                    "magnitudes belong to this cohort. The ordering and the procedure are what transfer.", {})],
             space_after=0),
    ], size=12, line_spacing=1.08)
    caption(s, LM, Inches(6.55), Inches(8.3),
            "Table 8 of main_v8.tex, redrawn. Sources: modality_and_no_chewing_…cead62e4, "
            "baselines_emg_only_20260816, outputs/screening/emg_only_*.json", size=9)

    # ---------------------------------------------------------------- advice
    s = deck.slide("advice", "What we tell readers",
                   "Audit your pipeline before you compare models — the five-step procedure", NOTES["advice"])
    steps = [
        ("1", "Audit the signal against what the analysis assumes",
         "Draw every filter response over the measured spectrum of your recordings. Fingerprint every channel of "
         "every recording. Publish the interference statistic beside the accuracy."),
        ("2", "Fix the data, vary one stage",
         "Same windows, same folds, same labels in every arm — verified ledger to ledger — then change "
         "one stage at a time."),
        ("3", "Price the choices made before training",
         "Window length, temporal context, normalisation scope, label source. Here two of them were worth more "
         "than the architecture."),
        ("4", "Report a budget beside the accuracy",
         "One scale, one row per stage, each naming its harness and its baseline. Say what is not additive."),
        ("5", "Keep a prediction ledger",
         "Every number and figure regenerated from saved held-out predictions, so a figure cannot disagree with "
         "the table it depicts."),
    ]
    cw5 = (12.133 - 0.15 * 4) / 5
    for i, (num, title, body) in enumerate(steps):
        x = LM + Inches((cw5 + 0.15) * i)
        add_rect(s, x, Inches(1.6), Inches(cw5), Inches(3.85), fill=WASH, rounded=True, radius=0.05)
        add_text(s, x + Inches(0.15), Inches(1.65), Inches(1.0), Inches(0.7),
                 [dict(text=num, size=30, bold=True, color=ACCENT)], space_after=0)
        add_text(s, x + Inches(0.15), Inches(2.35), Inches(cw5 - 0.3), Inches(0.95),
                 [dict(text=title, size=13, bold=True, color=INK)], space_after=0, line_spacing=1.02)
        add_text(s, x + Inches(0.15), Inches(3.3), Inches(cw5 - 0.3), Inches(2.1),
                 [dict(text=body, size=11, color=SLATE)], line_spacing=1.06)
    add_rect(s, LM, Inches(5.65), CW, Inches(1.2), fill=INK, rounded=True, radius=0.08)
    add_text(s, LM + Inches(0.25), Inches(5.65), CW - Inches(0.5), Inches(1.2), [
        dict(runs=[("Guards that refuse rather than warn: ", {"bold": True, "color": WHITE}),
                   ("the leakage assertion on every fold · a branch may not read a band its own filter deleted "
                    "· a run may not train on a held-out participant’s own waveform.", {"color": WHITE})],
             space_after=4),
        dict(runs=[("Cost: one pass over the data. ", {"bold": True, "color": WHITE}),
                   ("Both of our defects had the same shape — a correct check scoped one step too narrowly: to the "
                    "filter rather than the data, to the file rather than the dataset.", {"color": WHITE})],
             space_after=0),
    ], size=12, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.06)

    # ---------------------------------------------------------------- group / NIH
    s = deck.slide("group", "What it means for the group",
                   "The EMG-only paper is stronger, and the audit helps the NIH direction", NOTES["group"])
    colw = Inches(5.95)
    x2 = LM + colw + Inches(0.25)
    header_line(s, LM, Inches(1.55), colw, "Why the reframe is the right call")
    add_text(s, LM, Inches(1.9), colw, Inches(3.3), [
        dict(text="A LOSO number on a channel that was not held out is the kind of result that gets papers "
                  "corrected; a caveat cannot fix a mislabelled number", bullet=True),
        dict(text="The EMG paper has a clear question, a quantified answer, and a method other groups can adopt "
                  "in one pass over their own data", bullet=True),
        dict(text="Two independent measurement-chain defects found and fixed in one dataset — with the checks "
                  "that catch them — is a contribution, not a confession", bullet=True),
        dict(text="The v8 draft is complete: 26 pp, references from p. 20, every number traced to a saved artifact",
             bullet=True),
    ], size=12, space_after=6)
    header_line(s, x2, Inches(1.55), colw, "Why the audit strengthens a mic + EMG grant", color=GOOD)
    add_text(s, x2, Inches(1.9), colw, Inches(3.3), [
        dict(text="Nothing here refutes the acoustic hypothesis: these files contain no usable audio, so it is "
                  "untested", bullet=True),
        dict(text="It removes a latent leakage artifact before it could sit in a Preliminary Studies section",
             bullet=True),
        dict(text="Concrete rigor-and-reproducibility evidence: a measurement, a versioned policy, automated "
                  "tests, an ingest gate that refuses flagged data", bullet=True),
        dict(text="It turns “we will collect audio” into a protocol-plus-validation aim; the nine-requirement "
                  "spec is already written (Code/docs/audio_collection_spec.md)", bullet=True),
    ], size=12, space_after=6)
    add_text(s, LM, Inches(5.05), CW, Inches(0.3),
             [dict(text="PROPOSAL", size=10.5, bold=True, color=ACCENT)], space_after=0)
    chips = [
        ("1", "Publish the EMG-only pipeline paper now — v8 is drafted"),
        ("2", "Audio re-collection pilot, n ≈ 5–10, against the spec: ≥ 16 kHz WAV on its own clock, "
              "hardware sync marker, fingerprint check at ingest"),
        ("3", "The pilot becomes the preliminary data for the grant’s audio aim — with the failure "
              "analysis as the justification"),
    ]
    cw3 = (12.133 - 0.3) / 3
    for i, (n_, t) in enumerate(chips):
        x = LM + Inches((cw3 + 0.15) * i)
        add_rect(s, x, Inches(5.4), Inches(cw3), Inches(1.4), fill=WASH, rounded=True, radius=0.07)
        add_text(s, x + Inches(0.15), Inches(5.42), Inches(0.6), Inches(0.6),
                 [dict(text=n_, size=24, bold=True, color=ACCENT)], space_after=0)
        add_text(s, x + Inches(0.7), Inches(5.45), Inches(cw3 - 0.85), Inches(1.3),
                 [dict(text=t, size=11.5, color=INK)], anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.05)

    # ---------------------------------------------------------------- journals
    s = deck.slide("journals", f"Venues 1 of 2 — journals · impact factors JCR 2025 · checked {CHECKED}",
                   "Where to submit: journals first", NOTES["journals"])
    rows = [["Journal", "JIF", "Model · fee", "Speed (publisher-stated)", "Why it fits", "Read"]]
    rows += [list(r) for r in JOURNALS]
    add_table(s, LM, Inches(1.5), [2.95, 0.7, 1.25, 1.75, 3.95, 1.53], rows, size=9.5, row_h=0.4,
              col_align=[None, PP_ALIGN.CENTER, None, None, None, PP_ALIGN.CENTER], bold_first_col=True)
    add_text(s, LM, Inches(5.25), CW, Inches(0.8), [dict(text=JOURNALS_ALSO, size=9.5, color=SLATE)],
             line_spacing=1.03, space_after=0)
    add_rect(s, LM, Inches(6.1), CW, Inches(0.8), fill=WASH, rounded=True, radius=0.1)
    add_text(s, LM + Inches(0.2), Inches(6.1), CW - Inches(0.4), Inches(0.8), [
        dict(runs=[("JIF ", {"bold": True}),
                   ("= Clarivate Journal Impact Factor, JCR 2025 (released 17 Jun 2026), cross-checked against "
                    "publisher pages where shown; CBM’s 2025 value was not yet listed at the check date. "
                    "Fees are 2026 open-access charges (IEEE: −5% members, −20% society members). "
                    "“n.s.” = not stated by the publisher.", {})])],
        size=9.5, color=SLATE, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.03)

    # ---------------------------------------------------------------- conferences
    s = deck.slide("conferences", f"Venues 2 of 2 — conferences · checked {CHECKED}",
                   "Conference calendar: none in New England; the US options and their deadlines",
                   NOTES["conferences"])
    rows = [["Conference", "Next edition", "Where", "Paper deadline", "Proceedings", "Against the three conditions"]]
    rows += [list(r) for r in CONFERENCES]
    add_table(s, LM, Inches(1.5), [2.55, 2.15, 1.45, 2.45, 1.45, 2.08], rows, size=9, row_h=0.272,
              bold_first_col=True)
    add_rect(s, LM, Inches(6.28), CW, Inches(0.66), fill=WASH, rounded=True, radius=0.1)
    add_text(s, LM + Inches(0.2), Inches(6.28), CW - Inches(0.4), Inches(0.66), [
        dict(runs=[("Conditions: US · New England · full papers in proceedings. ", {"bold": True}),
                   ("None announced for 2026–27 meets all three. ", {"bold": True, "color": BAD}),
                   ("“Expected” deadlines are inferred from the previous edition and must be confirmed against the "
                    "2027 call. Also checked and outside the US or region: BSN ’26 Porto, BioCAS ’26 Incheon, "
                    "ICHI ’26 Minneapolis, CHIL ’26 Seattle, MLSP ’26 Atlanta, BIBM ’26 Dallas, ICMLA ’26 "
                    "Michigan, CBMS ’26 Cyprus, SenSys ’26 Saint-Malo, UbiComp ’26 Shanghai, BIOSTEC ’27 Malta, "
                    "HPEC ’26 Boston (HPC), URTC ’26 Cambridge (undergraduates only).", {})])],
        size=9, color=SLATE, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.03)

    # ---------------------------------------------------------------- decisions
    s = deck.slide("decisions", "Decisions", "What I need from this meeting", NOTES["decisions"])
    items = [
        ("1", "Approve the v8 framing and title",
         "A general stage-by-stage analysis with EMG jaw activity as the case study. 26 pp, references from p. 20; "
         "every number traces to a saved artifact."),
        ("2", "Pick the target journal",
         "Recommendation: IEEE JBHI (JIF 7.7). OJEMB (3.1) or IEEE Access (4.2) if being citable before the grant matters more than the venue."),
        ("3", "Approve the audio re-collection pilot and its acceptance criteria",
         "The grant’s audio aim rests on it; the nine-requirement spec is written and each requirement traces to an observed failure."),
        ("4", "Close the open items before submission",
         "Hardware, gain and ADC documentation (Q3) · the IRB identifier (records carry conflicting values, Q10) "
         "· data-release scope (Q11) · co-author sign-off on the dropped microphone and clinical sections."),
    ]
    y = Inches(1.6)
    for n_, t, b in items:
        add_rect(s, LM, y, CW, Inches(1.08), fill=WASH, rounded=True, radius=0.06)
        add_text(s, LM + Inches(0.2), y + Inches(0.12), Inches(0.7), Inches(0.8),
                 [dict(text=n_, size=30, bold=True, color=ACCENT)], space_after=0)
        add_text(s, LM + Inches(0.95), y + Inches(0.1), CW - Inches(1.2), Inches(0.4),
                 [dict(text=t, size=15, bold=True, color=INK)], space_after=0)
        add_text(s, LM + Inches(0.95), y + Inches(0.5), CW - Inches(1.2), Inches(0.55),
                 [dict(text=b, size=11.5, color=SLATE)], space_after=0, line_spacing=1.03)
        y += Inches(1.18)
    add_text(s, LM, Inches(6.4), CW, Inches(0.5), [
        dict(text="Nothing in the EMG results changes if the answers are “yes”. Everything in the microphone "
                  "story changes if the pilot is not run.", size=13, bold=True, color=INK, align=PP_ALIGN.CENTER)])

    assert deck.n == deck.total == len(deck.order), (deck.n, deck.total)
    assert deck.order == list(SLIDE_TITLES), "slide order and SLIDE_TITLES disagree"
    prs.save(DECK)
    write_script(deck.order)
    return DECK


# ----------------------------------------------------------------------------- script
QA = """\
## Anticipated questions (from the 19 Aug briefing, plus three for the new framing)

**"Can't we just fix or realign the audio?"** No. The circular offset is known only *between copies*, never relative to that recording's own EMG, and the misalignment differs per recording (median best lag 18 s, nothing under 0.1 s). There is also nothing to realign to: the `.avi` files have no audio stream and the `.npy` companions have a zero microphone column.

**"Can we at least use S05, whose audio is mostly unique?"** S05 has 4 of 20 recordings duplicated, so leakage is milder there, but duplication is the *second* problem. S05's audio is the same sub-10-Hz, 1-count-step envelope as everyone else's, at the quantisation floor for clenching and grinding. One participant is also n = 1. It is a useful control (it is the worst participant on audio and the best on EMG, which is what leakage looks like), not a dataset.

**"Could we report the ablation with a strong caveat?"** We drafted it and rejected it. The caveat cannot rescue the number because the number is mislabelled rather than uncertain: it is presented as held-out and it is not. The bias is not even uniform in direction. It would be the one thing in the paper a careful reviewer could disqualify us for.

**"Does admitting this make the group look careless?"** The paper's whole argument is that summary accuracies hide where performance comes from, and that cheap measurements on your own data catch what design review does not. We found two independent measurement-chain defects and report both with the checks that catch them. The bad version of this story is the one where somebody else finds it.

**"How do we know the EMG is really clean?"** The same fingerprint test, on every EMG channel of every recording: 100 distinct waveforms out of 100 on all four channels, zero cross-participant sharing. It is asserted on every manifest build, so it cannot silently stop being true.

**"Is the trigger channel safe? Every label depends on it."** 95 distinct waveforms in 100 recordings; the five collisions are all-zero quiet-rest triggers, which is expected for a binary channel with no events. The trigger was written by the same acquisition software, and a targeted trigger audit is logged as an open item before the next collection.

**"Isn't the new paper just a negative result?"** No: the headline is a five-class result on unseen participants with a quiet-rest class, plus a measured budget of what each stage was worth. The filter finding is the first row of that budget, not the thesis; v6 tried the "measurement chain outweighs the model" framing and it was hard to place, which is why v7/v8 sell the whole pipeline.

**"Why claim generality from five participants?"** The claim is for the *procedure*, never the numbers. The paper says explicitly that a lab with clean mains would find the first row worth nothing, that magnitudes belong to this cohort, and that whether the ordering recurs elsewhere is a question the study poses and does not answer.

**"The extended model beats the reference. Why not promote it?"** A 2.8-point mean difference resting on three of five participants (and losing 6.7 on P2) is a direction, not an effect size; the compact model delivers 96% of the macro-F1 at 33% of the parameters and 46% of the latency; and the two are one family, so the gap is evidence *for* the representational argument. Promotion remains available if a reviewer pushes; the checkpoints exist.

**"Why is the impact factor for Computers in Biology and Medicine dated 2024?"** Its JCR 2025 value had not been posted by the aggregator or the publisher page at the check date; the 2024 value (6.3) is the latest confirmed one. Check Clarivate directly before citing it.
"""

ASSUMPTIONS = """\
## Assumptions made while building this deck

- **Audience:** the co-author group and the advisor, i.e. people who know the project but have not read v8. The deck therefore spends six slides on the microphone evidence (the advisor briefing's figures, in its order) and one on the NIH argument.
- **"Version 2 and 3":** read as the original `Main_2.tex` (28 Jul, the Scientific Reports resubmission draft with `\\TBD` placeholders) and the current `Main_2.tex` (9 Aug, same direction with real numbers). The v1 claims quoted (85.0% / 80.3% / +17.4) are those the reviewers saw; an earlier draft in `misc/` reported 81.8% / 80.0% / +14.2.
- **Journal named in the request:** there is no "IEEE Open Journal of Computers in Medicine and Biology"; the closest venues are IEEE OJEMB (Open Journal of Engineering in Medicine and Biology) and Elsevier's *Computers in Biology and Medicine*. Both are on the journals slide.
- **Impact factors** are Clarivate JCR 2025 values (released 17 Jun 2026) as reported by publisher pages where those render (IOP, Wiley, Springer, IEEE Sensors Council, IEEE Access) and by a year-by-year JCR aggregator for the rest; the OpenAlex-based "Journal Metrics Score" that some sites label as an impact factor was *not* used. Verify in JCR before quoting in a cover letter.
- **Conference conditions** (US, New England, full papers published): applied strictly. Nothing announced meets all three as of 28 Aug 2026, so the calendar slide says so; "expected" deadlines are inferred from the previous edition and are flagged as such.
- **Pronouns:** the advisor is referred to by role throughout.
"""


def write_script(order):
    lines = [
        "# Script — “Where Does the Performance Come From?” (scope update, August 2026)",
        "",
        f"Deck: `scope_update_deck.pptx` ({len(order)} slides, 16:9). The same text is in each slide’s notes pane. "
        "Planned length about 24 minutes plus discussion; the per-slide budget is in brackets.",
        "",
        "Every number is quoted from `main_v8.tex` (26 Aug), `advisor/briefing.tex` (19 Aug), `cause.md` / `audio.md` "
        "(git HEAD) and the Scientific Reports decision letter (`Paper/Reviews/`). Venue facts were checked on "
        f"{CHECKED}; sources are in `venues.md`. Nothing was recomputed for the deck.",
        "",
        "## Running order",
        "",
    ]
    for i, k in enumerate(order, 1):
        lines.append(f"{i}. {SLIDE_TITLES[k]} [{TIMING[k]}]")
    lines += ["", "---", ""]
    for i, k in enumerate(order, 1):
        lines += [f"## Slide {i} — {SLIDE_TITLES[k]}  [{TIMING[k]}]", "", NOTES[k].strip(), "", "---", ""]
    lines += [QA, "---", "", ASSUMPTIONS]
    SCRIPT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    out = build()
    print(f"wrote {out}")
    print(f"wrote {SCRIPT}")
