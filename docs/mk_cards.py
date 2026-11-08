import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch
from PIL import Image
plt.rcParams["font.family"] = "DejaVu Sans"

def cards(out, icons_dir, items, cols, title=None, accent="#2F7BE5"):
    rows = (len(items) + cols - 1) // cols
    W, H = cols * 3.0, rows * 1.55 + (0.45 if title else 0)
    fig = plt.figure(figsize=(W, H), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    top = H - (0.45 if title else 0)
    if title:
        ax.text(0.15, H - 0.25, title, fontsize=13, fontweight="bold", color="#1F2328", va="center")
    for k, (icon, big, small, col) in enumerate(items):
        r, c = divmod(k, cols)
        x0, y0 = c * 3.0 + 0.1, top - (r + 1) * 1.55 + 0.1
        ax.add_patch(FancyBboxPatch((x0, y0), 2.8, 1.35, boxstyle="round,pad=0,rounding_size=0.18",
                                    fc="white", ec=col, lw=1.6))
        ax.add_patch(FancyBboxPatch((x0, y0), 0.12, 1.35, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=col, ec="none"))
        img = Image.open(f"{icons_dir}/{icon}.png").convert("RGBA")
        ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.17 * 256 / img.size[0]),
                                     (x0 + 0.58, y0 + 0.68), frameon=False))
        ax.text(x0 + 1.02, y0 + 0.86, big, fontsize=17, fontweight="bold", color=col, va="center")
        ax.text(x0 + 1.02, y0 + 0.40, small, fontsize=8.0, color="#57606A", va="center", linespacing=1.15)
    fig.savefig(out, facecolor="white"); plt.close(fig)

B, O, G, P, S_, R = "#2F7BE5", "#F07338", "#19A974", "#7B5CD6", "#5B6573", "#E5484D"
root = sys.argv[1]
cards(f"{root}/label/docs/figures/data_at_a_glance.png", f"{root}/label/paper/icons", [
    ("card_file_box", "8.1M", "PrimeKG rows\n(biomedical knowledge graph)", P),
    ("anatomical_heart", "9,388", "indication edges\n(what a drug treats)", B),
    ("warning", "30,675", "contraindication edges\n(who must not take it)", O),
    ("pill", "574", "drugs in the matched sample\n3 + 3 facts per drug", S_),
    ("globe_with_meridians", "3", "text sources: Wikipedia,\nPubMed, FDA drug label", G),
    ("robot", "7", "language models\n0.5B to 14B, 3 families", R),
], cols=3, title="LABEL at a glance")
cards(f"{root}/clade/docs/figures/data_at_a_glance.png", f"{root}/clade/paper/icons", [
    ("pill", "856", "PrimeKG questions\n(drugs and diseases)", B),
    ("dna", "900", "GloBI questions\n(species interactions)", G),
    ("card_file_box", "17,801", "MONDO disease\nfamily links", P),
    ("file_cabinet", "77,852", "GloBI taxonomy links\n(species to genus to family)", G),
    ("robot", "5", "language models\n0.5B to 14B, 3 families", O),
    ("puzzle_piece", "3", "page types: answer named,\nfamily only, or silent", S_),
], cols=3, title="CLADE at a glance")
print("ok")
