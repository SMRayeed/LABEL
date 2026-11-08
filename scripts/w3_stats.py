#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistics reviewers asked for on the W3PHIAI model side. No model runs.

  1. Accuracy per relation and model, with 95% Wilson intervals.
  2. Is the gain from scale smaller for contraindications than for
     indications? Each edge is scored by both models, so the gain is paired
     per edge; the interval is a bootstrap over DRUGS, so edges of the same
     drug move together.
  3. Does the indication-over-contraindication gap survive different
     distractors? Reads the abstention test's closed-book rows, whose
     options are siblings from the answer's own disease family plus a
     "Not enough information" option, across up to four models.

    python3 w3_stats.py \\
        --small p1_edges_d2_Qwen2.5-0.5B-Instruct.csv \\
        --large p1_edges_d2_Qwen2.5-7B-Instruct.csv \\
        --abstain abstain_Qwen2.5-0.5B-Instruct_clean.csv \\
                  abstain_Qwen2.5-7B-Instruct_clean.csv \\
                  abstain_OLMo-2-1124-7B-Instruct_clean.csv \\
                  abstain_granite-3.1-8b-instruct_clean.csv

Standard library only.
"""

import argparse
import csv
import math
import os
import random
from collections import defaultdict


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def pnorm(z):
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def load(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--abstain", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sm, lg = load(a.small), load(a.large)
    rels = sorted({r["relation"] for r in sm})

    print("=" * 78)
    print("1. ACCURACY PER RELATION, five options, no page, chance 20%")
    print(f"    {'relation':<18} {'model':<7} {'n':>5}   accuracy [95% CI]")
    for rel in rels:
        for name, rows in (("small", sm), ("large", lg)):
            sel = [int(r["mc_correct"]) for r in rows if r["relation"] == rel]
            p, lo, hi = wilson(sum(sel), len(sel))
            print(f"    {rel:<18} {name:<7} {len(sel):>5}   "
                  f"{p:5.2f}% [{lo:5.2f}, {hi:5.2f}]")

    # ------------------------------------------------- paired gain per edge
    s_by = {r["id"]: r for r in sm}
    gains = defaultdict(list)            # relation -> [(drug, gain)]
    for r in lg:
        s = s_by.get(r["id"])
        if s is None:
            continue
        gains[r["relation"]].append(
            (r["source"], int(r["mc_correct"]) - int(s["mc_correct"])))
    print("\n" + "=" * 78)
    print("2. GAIN FROM SCALE, paired per edge, bootstrap over drugs")
    for rel in rels:
        g = [x for _, x in gains[rel]]
        if g:
            print(f"    {rel:<18} n={len(g):>5}   gain "
                  f"{100 * sum(g) / len(g):+6.2f} points")
    if "indication" in gains and "contraindication" in gains:
        ind, con = gains["indication"], gains["contraindication"]
        diff = (100 * sum(x for _, x in ind) / len(ind) -
                100 * sum(x for _, x in con) / len(con))
        drugs = sorted({d for d, _ in ind} | {d for d, _ in con})
        bi, bc = defaultdict(list), defaultdict(list)
        for d, x in ind:
            bi[d].append(x)
        for d, x in con:
            bc[d].append(x)
        rng = random.Random(a.seed)
        boots = []
        for _ in range(a.boot):
            pick = [rng.choice(drugs) for _ in drugs]
            si = [x for d in pick for x in bi[d]]
            sc = [x for d in pick for x in bc[d]]
            if si and sc:
                boots.append(100 * sum(si) / len(si) - 100 * sum(sc) / len(sc))
        boots.sort()
        lo = boots[int(0.025 * len(boots))]
        hi = boots[int(0.975 * len(boots))]
        se = (hi - lo) / (2 * 1.96)
        z = diff / se if se > 0 else 0.0
        share = sum(b <= 0 for b in boots) / len(boots)
        print(f"\n    indication gain minus contraindication gain: "
              f"{diff:+.2f} points  [{lo:+.2f}, {hi:+.2f}]")
        print(f"    z={z:+.2f}  p={pnorm(z):.2g}  "
              f"(bootstrap share at or below zero {share:.4f}; "
              f"{len(drugs):,} drugs resampled)")

    # --------------------------------------- the gap under other distractors
    if a.abstain:
        print("\n" + "=" * 78)
        print("3. THE GAP UNDER DIFFERENT DISTRACTORS")
        print("   closed book, options from the answer's own disease family "
              "plus 'Not enough information'")
        print(f"    {'model':<34} {'indication':>16} {'contraind.':>16} "
              f"{'gap':>7}      p")
        for path in a.abstain:
            if not os.path.exists(path):
                print(f"    skip {path} -- not found")
                continue
            rows = [r for r in load(path) if r["condition"] == "closed"]
            k, n = {}, {}
            for rel in ("indication", "contraindication"):
                sel = [r for r in rows if r["relation"] == rel]
                n[rel] = len(sel)
                k[rel] = sum(r["outcome"] == "correct" for r in sel)
            if not n["indication"] or not n["contraindication"]:
                continue
            p1 = k["indication"] / n["indication"]
            p2 = k["contraindication"] / n["contraindication"]
            pp = (k["indication"] + k["contraindication"]) / (
                n["indication"] + n["contraindication"])
            se = math.sqrt(pp * (1 - pp) * (1 / n["indication"] +
                                            1 / n["contraindication"]))
            z = (p1 - p2) / se if se else 0.0
            name = os.path.basename(path).replace("abstain_", "")
            name = name.replace("_clean.csv", "")
            print(f"    {name:<34} {100 * p1:6.1f}% (n={n['indication']:>3}) "
                  f"{100 * p2:6.1f}% (n={n['contraindication']:>3}) "
                  f"{100 * (p1 - p2):+6.1f}   {pnorm(z):.2g}")
            # the two relations have different mixes of named, family-only
            # and silent questions; this is the gap within each group,
            # combined with weights n1*n2/(n1+n2)
            num = den = var = 0.0
            for g in sorted({r.get("group", "") for r in rows}):
                gi = [r for r in rows if r.get("group", "") == g and
                      r["relation"] == "indication"]
                gc = [r for r in rows if r.get("group", "") == g and
                      r["relation"] == "contraindication"]
                if not gi or not gc:
                    continue
                qi = sum(r["outcome"] == "correct" for r in gi) / len(gi)
                qc = sum(r["outcome"] == "correct" for r in gc) / len(gc)
                w = len(gi) * len(gc) / (len(gi) + len(gc))
                num += w * (qi - qc)
                den += w
                var += w * w * (qi * (1 - qi) / len(gi) +
                                qc * (1 - qc) / len(gc))
            if den:
                d = num / den
                zs = d / (math.sqrt(var) / den) if var else 0.0
                print(f"    {'':<34} within naming group: "
                      f"{100 * d:+6.1f} points   p={pnorm(zs):.2g}")


if __name__ == "__main__":
    main()
