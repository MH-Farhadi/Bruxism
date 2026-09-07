"""Composite figures for main_v9 (OJEMB): multipart figures combined and labelled
before submission, as the template requires. Sources are the v8 figure files; no
data are redrawn here."""
from PIL import Image, ImageDraw, ImageFont
P = "/home/kye/Desktop/Depo/Code/Bruxism/Paper/K_Farhadi_Paper_Bruxism/Figures/"
OUT = P + "v9/"
FONT = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"

def flat(im):
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return bg.convert("RGB")

def fit_h(im, h):
    w = round(im.width * h / im.height)
    return im.resize((w, h), Image.LANCZOS)

def label(draw, text, cx, y, px):
    f = ImageFont.truetype(FONT, px)
    w = draw.textlength(text, font=f)
    draw.text((cx - w / 2, y), text, fill=(0, 0, 0), font=f)

# ---------- Fig. 1: (a) pipeline, (b) placement schematic, (c) placement photo ----------
pipe = flat(Image.open(P + "Pipeline_V2.png"))
rl4 = flat(Image.open(P + "RL4.png"))
schem = rl4.crop((40, 60, 1320, 1010))      # head schematic with electrode labels
photo = rl4.crop((492, 1536, 1308, 2330))   # real-world placement panel
H2 = 950
schem = fit_h(schem, H2); photo = fit_h(photo, H2)
W = pipe.width
gap, lab, pad = 120, 130, 60
row2_w = schem.width + gap + photo.width
canvas = Image.new("RGB", (W, pipe.height + lab + pad + H2 + lab), "white")
d = ImageDraw.Draw(canvas)
canvas.paste(pipe, (0, 0))
label(d, "(a)", W // 2, pipe.height + 20, 58)
y2 = pipe.height + lab + pad
x0 = (W - row2_w) // 2
canvas.paste(schem, (x0, y2))
canvas.paste(photo, (x0 + schem.width + gap, y2))
label(d, "(b)", x0 + schem.width // 2, y2 + H2 + 20, 58)
label(d, "(c)", x0 + schem.width + gap + photo.width // 2, y2 + H2 + 20, 58)
canvas.save(OUT + "fig1_pipeline_setup.png", dpi=(600, 600), optimize=True)
print("fig1", canvas.size, "aspect", round(canvas.width / canvas.height, 2),
      "height at 7.16in:", round(7.16 * canvas.height / canvas.width, 2), "in")

# ---------- Fig. S3: (a) ROC, (b) t-SNE ----------
roc = flat(Image.open(P + "v5/five_class_roc_curves.png"))
tsne = flat(Image.open(P + "v5/tsne_5class.png"))
H = tsne.height
roc = fit_h(roc, H)
gap, lab = 100, 110
canvas = Image.new("RGB", (roc.width + gap + tsne.width, H + lab), "white")
d = ImageDraw.Draw(canvas)
canvas.paste(roc, (0, 0)); canvas.paste(tsne, (roc.width + gap, 0))
label(d, "(a)", roc.width // 2, H + 15, 50)
label(d, "(b)", roc.width + gap + tsne.width // 2, H + 15, 50)
canvas.save(OUT + "figS_roc_tsne.png", dpi=(300, 300), optimize=True)
print("figS_roc_tsne", canvas.size, "height at 7.16in:", round(7.16 * canvas.height / canvas.width, 2), "in")
