"""Fig. 1 for main_v9 (OJEMB): (a) acquisition chain and sensor placement (Paper/Picture1.png,
drawn by Kye, 2026-09-07) over (b) the signal-processing and model diagram (Figures/Pipeline_V2.png).
Panel (b) receives the same rounded gray frame as panel (a) so the two read as one designed figure."""
from PIL import Image, ImageDraw, ImageFont

P = "/home/kye/Desktop/Depo/Code/Bruxism/Paper/"
FIG = P + "K_Farhadi_Paper_Bruxism/Figures/"
OUT = FIG + "v9/fig1_system_pipeline.png"
FONT = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
FRAME = (136, 137, 141)      # measured from Picture1.png: 5-px line, 12-px inset, ~34-px outer radius

def flat(im):
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return bg.convert("RGB")

def label(draw, text, cx, y, px):
    f = ImageFont.truetype(FONT, px)
    w = draw.textlength(text, font=f)
    draw.text((cx - w / 2, y), text, fill=(0, 0, 0), font=f)

a = Image.open(P + "Picture1.png").convert("RGB")           # 3840 x 2160, framed
W = a.width

# ---- panel (b): pipeline diagram, flattened, scaled to the same width, framed like (a) ----
pipe = flat(Image.open(FIG + "Pipeline_V2.png"))           # 3379 x 1881, transparent margins ~72 px
inset = 20
pw = W - 2 * inset
ph = round(pipe.height * pw / pipe.width)
pipe = pipe.resize((pw, ph), Image.LANCZOS)
b = Image.new("RGB", (W, ph + 2 * inset), "white")
b.paste(pipe, (inset, inset))
ImageDraw.Draw(b).rounded_rectangle([12, 12, W - 13, b.height - 13], radius=34, outline=FRAME, width=5)

# ---- stack: (a) / label / gap / (b) / label ----
LAB, GAP, FONTPX = 95, 40, 66        # 66 px = 8 pt at 0.92\textwidth (6.59 in) for a 3840-px-wide figure
canvas = Image.new("RGB", (W, a.height + LAB + GAP + b.height + LAB), "white")
d = ImageDraw.Draw(canvas)
canvas.paste(a, (0, 0))
label(d, "(a)", W // 2, a.height + 14, FONTPX)
yb = a.height + LAB + GAP
canvas.paste(b, (0, yb))
label(d, "(b)", W // 2, yb + b.height + 14, FONTPX)
canvas.save(OUT, dpi=(600, 600), optimize=True)
print("fig1", canvas.size, "aspect", round(canvas.width / canvas.height, 3),
      "| height at 0.92*7.16 in:", round(0.92 * 7.16 * canvas.height / canvas.width, 2), "in",
      "| at 7.16 in:", round(7.16 * canvas.height / canvas.width, 2), "in")
