#!/usr/bin/env python3
"""
Build a drug-matched edge set, so the within-drug test has power.

Why this is needed. The current sample draws 600 edges per relation
independently, so a drug lands in the indication sample in proportion to its
indication edges and in the contraindication sample in proportion to its
contraindication edges. Only 101 drugs end up in both, most of them with one
or two edges on one side, and the within-drug estimate is too noisy to say
anything: Mantel-Haenszel returns 0.74 with an interval from 0.38 to 1.46.

Meanwhile the degree-adjusted pooled estimate is 2.27, which sits outside
that interval. Those two numbers are measuring different things. The pooled
one compares indication edges on one set of drugs against contraindication
edges on a largely different set; the within-drug one compares both on the
same drug. When they disagree, the within-drug estimate is the one that is
not confounded by which drugs ended up where.

This script fixes the sampling rather than the statistics. It keeps only
drugs carrying at least --min of each relation, then draws the SAME number
of edges of each relation from each drug. Every drug then contributes a
balanced 2x2 table, and drug identity cancels exactly.

    python3 drug_matched_sample.py \\
        --edges pkg_ind_d2d.csv pkg_con_d2dis.csv \\
        --a indication --b contraindication --min 3 --per-drug 3 \\
        --out-prefix matched

    # then re-measure, which mostly hits the Wikipedia cache already on disk
    python3 gonogo_p1.py build --edges matched_indication.csv matched_contraindication.csv \\
        --col-source x_name --col-target y_name --col-relation display_relation \\
        --names redirects --any-names --ancestors mondo_child_parent.csv \\
        --ancestor-depth 2 --n 2000 \\
        --out-edges matched_edges.csv --out-prompts matched_prompts.jsonl

    python3 dilution_test.py --scored matched_edges.csv \\
        --edges pkg_ind_d2d.csv pkg_con_d2dis.csv

Standard library only.
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict

csv.field_size_limit(min(2**31 - 1, sys.maxsize))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges", nargs="+", required=True)
    ap.add_argument("--col-source", default="x_name")
    ap.add_argument("--col-target", default="y_name")
    ap.add_argument("--col-relation", default="display_relation")
    ap.add_argument("--a", default="indication")
    ap.add_argument("--b", default="contraindication")
    ap.add_argument("--min", type=int, default=3,
                    help="a drug must carry at least this many of EACH")
    ap.add_argument("--per-drug", type=int, default=3,
                    help="edges drawn per relation per drug; the same number "
                         "on both sides is what makes the table balanced")
    ap.add_argument("--max-drugs", type=int, default=0,
                    help="cap the number of drugs, 0 for no cap")
    ap.add_argument("--out-prefix", default="matched")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = defaultdict(lambda: defaultdict(list))
    header = None
    for path in a.edges:
        if not os.path.exists(path):
            sys.exit(f"missing: {path}")
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            rd = csv.DictReader(fh)
            header = header or rd.fieldnames
            for need in (a.col_source, a.col_target, a.col_relation):
                if need not in (rd.fieldnames or []):
                    sys.exit(f"{path} has no column {need}. "
                             f"Columns: {', '.join(rd.fieldnames or [])}")
            for r in rd:
                src = (r.get(a.col_source) or "").strip()
                rel = (r.get(a.col_relation) or "").strip()
                if src and rel in (a.a, a.b):
                    rows[src][rel].append(r)

    both = [d for d, v in rows.items()
            if len(v.get(a.a, [])) >= a.min and len(v.get(a.b, [])) >= a.min]
    print(f"drugs carrying at least {a.min} of each relation: {len(both):,}")
    if not both:
        sys.exit("none qualify. Lower --min.")

    rng = random.Random(a.seed)
    both.sort()
    rng.shuffle(both)
    if a.max_drugs:
        both = both[:a.max_drugs]

    out = {a.a: [], a.b: []}
    for d in both:
        for rel in (a.a, a.b):
            pool = rows[d][rel]
            k = min(a.per_drug, len(pool))
            out[rel].extend(rng.sample(pool, k))

    for rel, rs in out.items():
        path = f"{a.out_prefix}_{rel.replace(' ', '_')}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            w.writerows(rs)
        print(f"  wrote {path}: {len(rs):,} edges over {len(both):,} drugs")

    n_each = min(len(out[a.a]), len(out[a.b]))
    print(f"\nEvery drug now contributes {a.per_drug} edges of each relation, so")
    print("the within-drug tables are balanced and drug identity cancels.")
    print(f"Expected Mantel-Haenszel strata: {len(both):,}  "
          f"(was 101, most of them with one edge on one side)")
    print(f"\nPower note: at {n_each:,} edges per relation and rates near "
          f"20% and 7%,")
    print("the within-drug test detects an odds ratio of about 1.6 or larger.")
    print("If it still returns a null at this size, the between-drug gap is")
    print("not a relation effect and the paper needs a different claim.")


if __name__ == "__main__":
    main()
