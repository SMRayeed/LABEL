#!/usr/bin/env python3
"""Split the PrimeKG edge export (kg.csv) into the files the pipeline reads.

PrimeKG stores every edge twice, once in each direction. Only the
drug -> disease direction is kept for the three drug-disease relations, and
the disease_disease rows are kept whole for mondo_parents.py, which reads
their x_id/x_name and y_id/y_name columns to map MONDO ids to PrimeKG's
disease names.

    python3 prep_primekg.py --kg kg.csv --outdir .

Writes
    pkg_ind_d2d.csv     indication        (drug -> disease)
    pkg_con_d2dis.csv   contraindication  (drug -> disease)
    pkg_off_d2dis.csv   off-label use     (drug -> disease)
    pkg_pc.csv          disease_disease rows (both directions, as stored)

Standard library only.
"""

import argparse
import csv
import os
import sys

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

OUT = {
    "indication": "pkg_ind_d2d.csv",
    "contraindication": "pkg_con_d2dis.csv",
    "off-label use": "pkg_off_d2dis.csv",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg", required=True)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    if not os.path.exists(a.kg):
        sys.exit(f"missing: {a.kg}")
    os.makedirs(a.outdir, exist_ok=True)

    with open(a.kg, newline="", encoding="utf-8", errors="replace") as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames or []
        for c in ("relation", "display_relation", "x_type", "x_name",
                  "y_type", "y_name", "x_id", "y_id"):
            if c not in cols:
                sys.exit(f"{a.kg} has no '{c}' column; columns: {cols}")
        handles, writers, counts = {}, {}, {}
        for rel, fname in list(OUT.items()) + [("disease_disease", "pkg_pc.csv")]:
            fh_out = open(os.path.join(a.outdir, fname), "w", newline="",
                          encoding="utf-8")
            handles[rel] = fh_out
            writers[rel] = csv.DictWriter(fh_out, fieldnames=cols)
            writers[rel].writeheader()
            counts[rel] = 0
        total = 0
        for r in rd:
            total += 1
            rel = (r.get("relation") or "").strip()
            disp = (r.get("display_relation") or "").strip()
            xt = (r.get("x_type") or "").strip()
            yt = (r.get("y_type") or "").strip()
            if rel == "disease_disease":
                writers[rel].writerow(r)
                counts[rel] += 1
                continue
            if xt == "drug" and yt == "disease" and disp in OUT:
                writers[disp].writerow(r)
                counts[disp] += 1
        for h in handles.values():
            h.close()

    print(f"{a.kg}: {total:,} rows")
    for rel, fname in list(OUT.items()) + [("disease_disease", "pkg_pc.csv")]:
        print(f"  {fname:<20} {counts[rel]:>9,}  {rel}")
    if counts["indication"] == 0:
        sys.exit("no indication edges found; is this the PrimeKG kg.csv?")


if __name__ == "__main__":
    main()
