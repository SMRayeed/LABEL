#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Promote one of the tier columns to `tier`, so the existing tests can read it.

second_corpus.py leaves three measurements side by side: tier_wiki,
tier_second and tier_union. Every test downstream reads a column called
`tier`, so a second-corpus measurement cannot be tested without a copy of
the file whose `tier` holds the column in question. This writes that copy.

The original column is kept under its own name as well, so a file that has
been retiered still records which measurement it carries and the step
cannot be mistaken for the first one later.

    python3 retier.py --in matched_edges_pm.csv --use tier_union \\
        --out matched_edges_union.csv

    python3 retier.py --in matched_edges_pm.csv --use tier_second \\
        --out matched_edges_pmonly.csv

Standard library only.
"""

import argparse
import csv
import os
import sys
from collections import Counter

csv.field_size_limit(min(2**31 - 1, sys.maxsize))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--use", required=True,
                    help="the column to promote, such as tier_union")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if not os.path.exists(a.src):
        sys.exit(f"missing: {a.src}")
    with open(a.src, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        cols = list(rd.fieldnames or [])
    if not rows:
        sys.exit(f"{a.src} has no rows")
    if a.use not in cols:
        sys.exit(f"{a.src} has no column {a.use!r}.\nColumns: "
                 + ", ".join(cols))

    out_cols = cols if "tier" in cols else ["tier"] + cols
    kept = 0
    for r in rows:
        v = (r.get(a.use) or "").strip()
        if v == "":
            continue
        r["tier"] = v
        kept += 1
    rows = [r for r in rows if (r.get("tier") or "").strip() != ""]

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    mix = Counter(r["tier"] for r in rows)
    print(f"wrote {a.out}: {kept:,} rows, tier taken from {a.use}")
    print("  tier mix " + "  ".join(f"{k}:{mix[k]:,}" for k in sorted(mix)))
    print(f"\nThe {a.use} values now sit in `tier`, so four_boxes.py, "
          f"tier_cliff.py and\ndilution_test.py read this file as they read "
          f"the first measurement.")


if __name__ == "__main__":
    main()
