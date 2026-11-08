#!/usr/bin/env python3
"""Collect the W3PHIAI headline numbers from the pipeline's CSV outputs.

Everything here is recomputed from the per-edge files, so the summary does
not depend on parsing the printed logs. The detailed tests (degree strata,
regressions, bootstrap) stay in the logs of dilution_test.py and w3_stats.py.

    python3 w3_summary.py --work work --small-tag Qwen2.5-0.5B-Instruct \
        --large-tag Qwen2.5-7B-Instruct --out results/SUMMARY.md

Standard library only.
"""

import argparse
import csv
import math
import os
from collections import defaultdict


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pchi1(x):
    return math.erfc(math.sqrt(max(x, 0.0) / 2.0))


def named(r, cut=2):
    try:
        return int(r["tier"]) <= cut
    except (KeyError, ValueError):
        return False


def rate(rows, rel, cut=2):
    v = [r for r in rows if r["relation"] == rel]
    if not v:
        return float("nan"), 0
    return 100.0 * sum(named(r, cut) for r in v) / len(v), len(v)


def mantel_haenszel(rows, a="indication", b="contraindication", cut=2):
    """Within-source MH odds ratio of being named, relation a vs b."""
    t = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        if r["relation"] not in (a, b):
            continue
        k = (0 if r["relation"] == a else 2) + (0 if named(r, cut) else 1)
        t[r["source"]][k] += 1
    R = S = 0.0
    PR = PS_QR = QS = 0.0
    obs = exp = var = 0.0
    strata = 0
    for x1, x2, y1, y2 in t.values():       # a named, a unnamed, b named, b unnamed
        n = x1 + x2 + y1 + y2
        if n < 2 or (x1 + x2) == 0 or (y1 + y2) == 0:
            continue
        strata += 1
        r_ = x1 * y2 / n
        s_ = x2 * y1 / n
        P = (x1 + y2) / n
        Q = (x2 + y1) / n
        R += r_
        S += s_
        PR += P * r_
        PS_QR += P * s_ + Q * r_
        QS += Q * s_
        n1, n2 = x1 + x2, y1 + y2
        m1, m2 = x1 + y1, x2 + y2
        obs += x1
        exp += n1 * m1 / n
        var += n1 * n2 * m1 * m2 / (n * n * (n - 1))
    if not (R and S) or not var:
        return None
    orr = R / S
    # Same interval and test as dilution_test.py, whose output the paper
    # reports: test-based standard error 1/sqrt(sum of variances) and the
    # continuity-corrected CMH chi-square.
    se = 1.0 / math.sqrt(var)
    chi = (abs(obs - exp) - 0.5) ** 2 / var
    # Robins-Breslow-Greenland interval, for comparison
    v = PR / (2 * R * R) + PS_QR / (2 * R * S) + QS / (2 * S * S)
    se_rbg = math.sqrt(v)
    return {"or": orr, "lo": math.exp(math.log(orr) - 1.96 * se),
            "hi": math.exp(math.log(orr) + 1.96 * se),
            "lo_rbg": math.exp(math.log(orr) - 1.96 * se_rbg),
            "hi_rbg": math.exp(math.log(orr) + 1.96 * se_rbg),
            "p": pchi1(chi), "strata": strata}


def raw_or(rows, a, b, cut=2):
    def cnt(rel):
        v = [r for r in rows if r["relation"] == rel]
        k = sum(named(r, cut) for r in v)
        return k, len(v)
    k1, n1 = cnt(a)
    k2, n2 = cnt(b)
    if not (n1 and n2) or k1 in (0, n1) or k2 in (0, n2):
        return None
    orr = (k1 / (n1 - k1)) / (k2 / (n2 - k2))
    p = (k1 + k2) / (n1 + n2)
    z = (k1 / n1 - k2 / n2) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return {"or": orr, "p": math.erfc(abs(z) / math.sqrt(2)),
            "r1": 100 * k1 / n1, "r2": 100 * k2 / n2, "n1": n1, "n2": n2}


def acc(rows, rel):
    v = [r for r in rows if r["relation"] == rel and r.get("mc_correct") not in (None, "")]
    if not v:
        return float("nan"), 0
    return 100.0 * sum(int(float(r["mc_correct"])) for r in v) / len(v), len(v)


def fmt_p(p):
    return "n/a" if p is None else (f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--small-tag", default="Qwen2.5-0.5B-Instruct")
    ap.add_argument("--large-tag", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="SUMMARY.md")
    a = ap.parse_args()
    W = lambda f: os.path.join(a.work, f)          # noqa: E731

    L = ["# W3PHIAI results summary", "",
         "Recomputed from the per-edge CSV files in `work/`. "
         "Detailed logs are in `results/`.", ""]

    # ---------------------------------------------------------- Table 1
    L += ["## Coverage of indications and contraindications (drug-matched sample)", "",
          "| Text searched | Indications named | Contraindications named | "
          "Within-drug OR [95% CI] | p | drugs | RBG 95% CI |",
          "|---|---|---|---|---|---|---|"]
    for label, f in (("Wikipedia", "matched_edges.csv"),
                     ("PubMed abstracts (balanced query)", "me_bal_pm.csv"),
                     ("Both together", "me_bal_union.csv"),
                     ("PubMed abstracts (single query)", "me_pm.csv"),
                     ("Wikipedia + PubMed (single query)", "me_union.csv")):
        rows = load(W(f))
        if not rows:
            L.append(f"| {label} | missing `{f}` | | | | | |")
            continue
        ri, ni = rate(rows, "indication")
        rc, nc = rate(rows, "contraindication")
        mh = mantel_haenszel(rows)
        if mh:
            L.append(f"| {label} | {ri:.2f}% (n={ni:,}) | {rc:.2f}% (n={nc:,}) | "
                     f"{mh['or']:.2f} [{mh['lo']:.2f}, {mh['hi']:.2f}] | "
                     f"{fmt_p(mh['p'])} | {mh['strata']:,} | "
                     f"[{mh['lo_rbg']:.2f}, {mh['hi_rbg']:.2f}] |")
        else:
            L.append(f"| {label} | {ri:.2f}% | {rc:.2f}% | n/a | | | |")
    L.append("")

    # ---------------------------------------------------------- probe
    small = load(W(f"p1_edges_d2_{a.small_tag}.csv"))
    large = load(W(f"p1_edges_d2_{a.large_tag}.csv"))
    L += ["## Closed-book probe (five options, chance 20%)", "",
          f"| Relation | {a.small_tag} | {a.large_tag} | Gain |", "|---|---|---|---|"]
    for rel in ("indication", "off-label use", "contraindication"):
        s, ns = acc(small, rel) if small else (float("nan"), 0)
        g, ng = acc(large, rel) if large else (float("nan"), 0)
        L.append(f"| {rel} | {s:.2f}% (n={ns}) | {g:.2f}% (n={ng}) | {g - s:+.2f} |")
    L += ["", "Paired bootstrap of the gain difference: see `results/w3_stats.txt`.", ""]

    # ---------------------------------------------------------- exact names
    p1 = load(W("p1_edges_d2.csv"))
    if p1:
        L += ["## Exact-name match in Wikipedia, probe sample", "",
              "| Relation | tier 0 | tiers 0-2 (named) | n |", "|---|---|---|---|"]
        for rel in ("indication", "off-label use", "contraindication"):
            r0, n = rate(p1, rel, cut=0)
            r2, _ = rate(p1, rel, cut=2)
            L.append(f"| {rel} | {r0:.2f}% | {r2:.2f}% | {n} |")
        L.append("")

    # ---------------------------------------------------------- Hetionet
    het = load(W("het_edges.csv"))
    L += ["## Negative control: Hetionet treats vs palliates (Wikipedia)", ""]
    if het:
        ro = raw_or(het, "treats", "palliates")
        mh = mantel_haenszel(het, "treats", "palliates")
        if ro:
            L.append(f"treats named {ro['r1']:.2f}% (n={ro['n1']}), palliates named "
                     f"{ro['r2']:.2f}% (n={ro['n2']}); odds ratio {ro['or']:.2f}, "
                     f"p = {fmt_p(ro['p'])}.")
        if mh:
            L.append(f"Within-drug MH odds ratio {mh['or']:.2f} "
                     f"[{mh['lo']:.2f}, {mh['hi']:.2f}], p = {fmt_p(mh['p'])}, "
                     f"{mh['strata']} drugs.")
    else:
        L.append("missing `het_edges.csv`")
    L.append("")

    L += ["## Where the other numbers are", "",
          "| Paper item | File |", "|---|---|",
          "| Within-drug tests and degree strata, Wikipedia | `results/dilution_wiki.txt` |",
          "| Within-drug tests and degree strata, PubMed | `results/dilution_pubmed.txt` |",
          "| Within-drug tests and degree strata, both corpora | `results/dilution_union.txt` |",
          "| Hetionet control, full test | `results/dilution_hetionet.txt` |",
          "| Wilson intervals, paired gain from scale, harder-distractor check | `results/w3_stats.txt` |",
          "| Silent share per corpus, single and balanced PubMed query | `results/second_corpus_*.txt` |",
          "| Floor and contribution of each naming level | `results/name_levels_depth*.txt` |",
          "| MONDO table size | `results/mondo_parents.txt` |", ""]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
