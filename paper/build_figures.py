#!/usr/bin/env python3
"""Figures for the W3PHIAI paper. Every number is copied from the run
outputs (results/SUMMARY.md, results/EXTRAS.md, results/label_coverage.txt,
results/dilution_*.txt); nothing is recomputed here except Wilson intervals
from the reported accuracies and n = 600.

    python make_figures.py          # writes the PDFs next to this file
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
LBLUE, LORANGE, LGREY = "#e9f1fb", "#fdeee6", "#f3f2ee"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2,
    "ytick.color": INK2, "axes.labelcolor": INK, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
W = 3.3            # AAAI column width, inches
WW = 2 * W + 0.33  # full width
HERE = os.path.dirname(os.path.abspath(__file__))


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def wilson(p, n=600, z=1.96):
    k = round(p * n / 100)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def box(ax, x, y, w, h, fc=LGREY, ec=INK2, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw))


def arrow(ax, x1, y1, x2, y2, col=INK2, lw=0.8, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, linestyle=ls,
                                shrinkA=0, shrinkB=0))


def paired_bars(ax, groups, ind, con, ylab, ymax, notes=None, note_y=None):
    x = range(len(groups))
    bw = 0.36
    ax.bar([i - bw / 2 - 0.01 for i in x], ind, bw, color=BLUE, label="Indication")
    ax.bar([i + bw / 2 + 0.01 for i in x], con, bw, color=ORANGE, hatch="////",
           edgecolor="white", linewidth=0, label="Contraindication")
    for i in x:
        ax.text(i - bw / 2 - 0.01, ind[i] + 0.6, f"{ind[i]:.1f}", ha="center", va="bottom",
                fontsize=6.8, color=INK)
        ax.text(i + bw / 2 + 0.01, con[i] + 0.6, f"{con[i]:.1f}", ha="center", va="bottom",
                fontsize=6.8, color=INK)
        if notes and notes[i]:
            ax.text(i, max(ind[i], con[i]) + (note_y or 4.2), notes[i], ha="center",
                    fontsize=6.8, color=INK2, style="italic")
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylab)
    ax.set_ylim(0, ymax)
    grid(ax)
    ax.legend(loc="lower center", frameon=False, ncol=2, handlelength=1.2,
              bbox_to_anchor=(0.5, 1.0))


# ---------------------------------------------------------------- figures
def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(WW, 1.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 27)
    ax.axis("off")

    def b(x, y, w, h, title, sub, fc, ec):
        box(ax, x, y, w, h, fc, ec)
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=7.3,
                color=INK, weight="bold")
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=6.4,
                color=INK2, linespacing=1.05)

    b(0.5, 9, 14, 10, "Same edges", "574 drugs, PrimeKG\nindication vs.\ncontraindication",
      "white", INK2)
    ax.text(18, 25.4, "TEXT SIDE: is the fact written down?", fontsize=6.8, color=BLUE,
            weight="bold")
    ax.text(18, 0.2, "MODEL SIDE: does the model know it?", fontsize=6.8, color=ORANGE,
            weight="bold")
    top = [("Three corpora", "Wikipedia, PubMed,\nFDA label (openFDA)"),
           ("Match names", "exact, alternative,\nfamily only"),
           ("Chance floor", "same match on 3\nunrelated diseases")]
    bot = [("Closed book", "5 options, chance 20%\nQwen2.5 0.5B to 14B"),
           ("Harder options", "MONDO siblings,\n4 models, 3 families"),
           ("With the page", "same questions,\npassage in the prompt")]
    xs = [18, 36, 54]
    for x, (t, s_) in zip(xs, top):
        b(x, 15.2, 15.5, 8.6, t, s_, LBLUE, BLUE)
    for x, (t, s_) in zip(xs, bot):
        b(x, 3.2, 15.5, 8.6, t, s_, LORANGE, ORANGE)
    for a_, c_ in ((33.8, 35.7), (51.8, 53.7)):
        arrow(ax, a_, 19.5, c_, 19.5, BLUE)
        arrow(ax, a_, 7.5, c_, 7.5, ORANGE)
    arrow(ax, 15, 16, 17.6, 19.5, BLUE)
    arrow(ax, 15, 12, 17.6, 7.5, ORANGE)
    b(75, 6.5, 24.5, 14, "Compare within drug",
      "Mantel–Haenszel OR,\ndegree-matched strata,\npaired gain from scale,\ncontrol graph (Hetionet)",
      "white", INK)
    arrow(ax, 69.8, 19.5, 74.7, 16, BLUE)
    arrow(ax, 69.8, 7.5, 74.7, 11, ORANGE)
    fig.tight_layout(pad=0.2)
    fig.savefig(out)


def fig_coverage(out):
    fig, ax = plt.subplots(figsize=(W, 1.95))
    paired_bars(ax, ["Wikipedia", "PubMed\nabstracts", "Both\ntogether"],
                [20.33, 26.31, 29.04], [9.81, 11.44, 15.91], "Facts named in text (%)", 36,
                ["OR 2.43", "OR 2.77", "OR 2.19"])
    fig.tight_layout(pad=0.3)
    fig.savefig(out)


def fig_label(out):
    fig, ax = plt.subplots(figsize=(W, 2.15))
    groups = ["Wikipedia", "Wiki +\nPubMed", "Label,\nown\nsection", "Label,\nwhole", "All\nthree"]
    paired_bars(ax, groups, [22.25, 30.51, 19.07, 23.73, 34.04],
                [10.95, 17.23, 2.61, 22.88, 26.69], "Facts named (%), 472 drugs", 44,
                [None, None, "OR 9.85", "OR 1.05", "OR 1.43"], note_y=4.0)
    ax.axvline(1.5, color=GRID, linewidth=0.8)
    ax.text(0.5, 41.5, "prose", ha="center", fontsize=7, color=INK2)
    ax.text(3.0, 41.5, "FDA label", ha="center", fontsize=7, color=INK2)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)


def fig_degree(out):
    strata = ["3–5", "6–12", "13+"]
    fig, ax = plt.subplots(figsize=(W, 1.8))
    x = range(3)
    for vals, col, m, lab in (([13.94, 6.04, 1.64], BLUE, "o", "Wikipedia"),
                              ([11.81, 6.90, 2.23], AQUA, "s", "Wikipedia + PubMed"),
                              ([-3.59, -6.46, -1.50], ORANGE, "D", "FDA label, whole")):
        ax.plot(x, vals, color=col, marker=m, linewidth=2, markersize=5,
                markeredgecolor="white", label=lab)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(strata)
    ax.set_xlabel("Edges per drug in the relation")
    ax.set_ylabel("Indication minus\ncontraindication (pts)")
    ax.set_ylim(-9, 17)
    grid(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)


def fig_scale(out):
    sizes = [0.5, 1.5, 3, 7, 14]
    rels = [("Indication", [31.50, 45.67, 45.33, 50.83, 49.83], BLUE, "o"),
            ("Off-label use", [28.50, 35.83, 32.67, 35.00, 35.83], AQUA, "s"),
            ("Contraindication", [22.17, 25.33, 24.17, 23.83, 27.17], ORANGE, "D")]
    fig, ax = plt.subplots(figsize=(W, 2.05))
    for name, ys, col, m in rels:
        lo = [wilson(v)[0] for v in ys]
        hi = [wilson(v)[1] for v in ys]
        ax.fill_between(sizes, lo, hi, color=col, alpha=0.13, linewidth=0)
        ax.plot(sizes, ys, color=col, linewidth=2, marker=m, markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.7, zorder=3, label=name)
    ax.axhline(20, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(0.55, 18.0, "chance (20%)", fontsize=6.8, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(["0.5B", "1.5B", "3B", "7B", "14B"])
    ax.minorticks_off()
    ax.set_xlabel("Qwen2.5-Instruct parameters (log scale)")
    ax.set_ylabel("Accuracy, closed book (%)")
    ax.set_ylim(15, 57)
    grid(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=7, ncol=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)


def fig_retrieval(out):
    groups = ["Page names\nthe disease", "Page names\nonly its family", "Page is\nsilent"]
    ind = [(11.1, 6.7, 15.8), (-8.2, -13.3, -3.1), (-11.3, -17.1, -6.0)]
    con = [(14.2, 7.3, 20.9), (0.3, -6.3, 6.9), (-8.9, -12.9, -5.1)]
    fig, ax = plt.subplots(figsize=(W, 1.95))
    bw = 0.36
    for k, (rows, col, lab, hatch) in enumerate(((ind, BLUE, "Indication", None),
                                                  (con, ORANGE, "Contraindication", "////"))):
        xs = [i + (k - 0.5) * (bw + 0.02) for i in range(3)]
        ax.bar(xs, [r[0] for r in rows], bw, color=col, hatch=hatch, edgecolor="white",
               linewidth=0, label=lab)
        ax.errorbar(xs, [r[0] for r in rows],
                    yerr=[[r[0] - r[1] for r in rows], [r[2] - r[0] for r in rows]],
                    fmt="none", ecolor=INK, elinewidth=0.7, capsize=2)
        for x_, r in zip(xs, rows):
            ax.text(x_, r[2] + 0.8 if r[0] >= 0 else r[1] - 0.8, f"{r[0]:+.1f}", ha="center",
                    va="bottom" if r[0] >= 0 else "top", fontsize=6.6, color=INK)
    ax.axhline(0, color=INK2, linewidth=0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(groups)
    ax.set_ylabel("Change in accuracy with\nthe page (pts, 95% CI)")
    ax.set_ylim(-22, 27)
    grid(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)


def fig_router(out):
    fig, ax = plt.subplots(figsize=(WW, 2.05))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 31)
    ax.axis("off")

    def b(x, y, w, h, title, sub, fc, ec, fs=7.2):
        box(ax, x, y, w, h, fc, ec)
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="center", fontsize=fs,
                color=INK, weight="bold")
        ax.text(x + w / 2, y + (h - 3.6) / 2, sub, ha="center", va="center", fontsize=6.3,
                color=INK2, linespacing=1.1)

    b(0.5, 11, 13.5, 9, "Medication\nquestion", "", "white", INK2)
    ax.text(7.25, 13.2, "“Can a patient with\nmyasthenia gravis take\nciprofloxacin?”",
            ha="center", va="center", fontsize=6.0, color=INK2, style="italic")
    b(17, 10.5, 15.5, 10, "Controller", "classify the relation:\nbenefit or safety?\nset the source plan",
      "white", INK, fs=7.4)
    arrow(ax, 14.2, 15.5, 16.8, 15.5)

    # benefit branch (top)
    b(37, 18.5, 19, 12, "Benefit route", "retrieve Wikipedia / PubMed;\nverify the disease is named\n(text names 29% of indications;\nthe page adds +11.1 pts)",
      LBLUE, BLUE)
    # safety branch (bottom)
    b(37, 0.5, 19, 12, "Safety route", "query DrugCentral / the KG;\nread the whole FDA label\n(prose names 17%; the whole\nlabel shows no gap, OR 1.05)",
      LORANGE, ORANGE)
    arrow(ax, 32.7, 18, 36.8, 24, BLUE)
    arrow(ax, 32.7, 13, 36.8, 6.5, ORANGE)

    b(61, 10.5, 16.5, 10, "Evidence check", "is the exact disease\nnamed, or only its\nfamily, or nothing?",
      "white", INK)
    arrow(ax, 56.2, 24, 60.8, 18, BLUE)
    arrow(ax, 56.2, 6.5, 60.8, 13, ORANGE)

    b(82, 19.5, 17.5, 10.5, "Answer", "cite the source\nand the matched\nsentence", "white", AQUA)
    b(82, 1.0, 17.5, 10.5, "Decline / escalate", "state which source\nis silent; refer to\nthe label or a pharmacist",
      "white", INK2)
    arrow(ax, 77.7, 18, 81.8, 24, AQUA)
    arrow(ax, 77.7, 13, 81.8, 6.5, INK2)
    ax.text(79.6, 23.3, "named", fontsize=6.0, color=AQUA, ha="right")
    ax.text(79.6, 7.6, "silent", fontsize=6.0, color=INK2, ha="right")
    fig.tight_layout(pad=0.2)
    fig.savefig(out)


def fig_failure(out):
    rows = [("Qwen2.5 0.5B", (36.0, 60.7, 3.3), (28.1, 71.4, 0.5)),
            ("Qwen2.5 7B", (56.0, 22.0, 22.0), (25.1, 24.1, 50.7)),
            ("OLMo 2 7B", (56.0, 34.0, 10.0), (33.0, 64.0, 3.0)),
            ("Granite 3.1 8B", (58.7, 28.7, 12.7), (12.8, 19.7, 67.5))]
    fig, ax = plt.subplots(figsize=(W, 2.1))
    ys, labels = [], []
    y = 0
    for name, ind, con in rows[::-1]:
        for rel, vals, yy in (("Contra.", con, y), ("Ind.", ind, y + 0.42)):
            left = 0
            for v, col, hatch in zip(vals, (AQUA, ORANGE, "#c9c7c1"), (None, None, None)):
                ax.barh(yy, v, height=0.38, left=left, color=col, hatch=hatch,
                        edgecolor="white", linewidth=0)
                if v >= 9:
                    ax.text(left + v / 2, yy, f"{v:.0f}", ha="center", va="center",
                            fontsize=6.2, color="white" if col != "#c9c7c1" else INK)
                left += v
            ys.append(yy)
            labels.append(f"{name}  {rel}" if rel == "Ind." else rel)
        y += 1.15
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of questions, closed book (%)")
    ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=AQUA, label="correct"),
                       Patch(color=ORANGE, label="wrong"),
                       Patch(color="#c9c7c1", label="not enough information")],
              loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False,
              fontsize=6.8, handlelength=1.2)
    ax.spines["left"].set_visible(False)
    grid(ax, "x")
    fig.tight_layout(pad=0.3)
    fig.savefig(out)


if __name__ == "__main__":
    # fig_pipeline.pdf and fig_router.pdf are exported from drawio/*.drawio
    for name, fn in (("fig_coverage.pdf", fig_coverage),
                     ("fig_label.pdf", fig_label), ("fig_degree.pdf", fig_degree),
                     ("fig_scale.pdf", fig_scale), ("fig_retrieval.pdf", fig_retrieval),
                     ("fig_failure.pdf", fig_failure)):
        fn(os.path.join(HERE, name))
    print("ok")
