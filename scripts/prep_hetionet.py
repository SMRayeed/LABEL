#!/usr/bin/env python3
"""
Turn Hetionet's SIF edge list into the per-relation CSVs the tier and
matching scripts read.

Hetionet stores edges as three columns of namespaced identifiers --
Compound::DB00014, Disease::DOID:0050742 -- and keeps the human-readable
names in a separate node file. Everything downstream works on names, so this
joins the two and writes one CSV per metaedge with the column headers the
other scripts already default to.

    # see what is in the file before asking for anything
    python3 prep_hetionet.py --edges hetionet/hetnet/tsv/hetionet-v1.0-edges.sif.gz \\
        --nodes hetionet/hetnet/tsv/hetionet-v1.0-nodes.tsv --inspect

    # the two drug-disease relations
    python3 prep_hetionet.py --edges hetionet/hetnet/tsv/hetionet-v1.0-edges.sif.gz \\
        --nodes hetionet/hetnet/tsv/hetionet-v1.0-nodes.tsv \\
        --metaedges CtD,CpD --prefix het

Writes het_CtD.csv and het_CpD.csv with columns x_name, y_name,
display_relation, so the existing --col-source / --col-target /
--col-relation defaults for PrimeKG work unchanged.

Standard library only.
"""

import argparse
import csv
import gzip
import io
import os
import sys
from collections import Counter, defaultdict

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

# Hetionet's abbreviations are opaque on sight, so name the ones that matter.
KNOWN = {
    "CtD": "treats", "CpD": "palliates", "CdG": "downregulates",
    "CuG": "upregulates", "CbG": "binds", "DaG": "associates",
    "DdG": "downregulates", "DuG": "upregulates", "DlA": "localizes",
    "DpS": "presents", "DrD": "resembles", "CrC": "resembles",
    "GiG": "interacts", "GpBP": "participates", "AeG": "expresses",
}


def opener(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8",
                                errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def load_nodes(path):
    """id -> (name, kind). Tolerates either a header or none."""
    names, kinds = {}, {}
    with opener(path) as fh:
        first = fh.readline()
        cols = first.rstrip("\n").split("\t")
        has_header = [c.strip().lower() for c in cols[:3]] == ["id", "name", "kind"]
        if not has_header and len(cols) >= 2:
            names[cols[0]] = cols[1]
            if len(cols) >= 3:
                kinds[cols[0]] = cols[2]
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            names[p[0]] = p[1]
            if len(p) >= 3:
                kinds[p[0]] = p[2]
    if not names:
        sys.exit(f"{path}: no nodes read. Is this the Hetionet node TSV?")
    return names, kinds


def iter_edges(path):
    """Yield (source_id, metaedge, target_id), skipping any header line."""
    with opener(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            if p[0].strip().lower() in ("source", "source_id"):
                continue          # a header, not an edge
            yield p[0], p[1], p[2]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges", required=True, help="hetionet-v1.0-edges.sif(.gz)")
    ap.add_argument("--nodes", required=True, help="hetionet-v1.0-nodes.tsv")
    ap.add_argument("--metaedges", default="CtD,CpD",
                    help="comma-separated, as they appear in the SIF")
    ap.add_argument("--prefix", default="het", help="output filename prefix")
    ap.add_argument("--inspect", action="store_true",
                    help="list every metaedge with its count and stop")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    for p in (a.edges, a.nodes):
        if not os.path.exists(p):
            sys.exit(f"no such file: {p}")

    if a.inspect:
        counts = Counter()
        kinds = defaultdict(Counter)
        for s, m, t in iter_edges(a.edges):
            counts[m] += 1
            kinds[m][(s.split("::")[0], t.split("::")[0])] += 1
        print(f"{len(counts)} metaedges in {a.edges}\n")
        print(f"  {'metaedge':<10}{'edges':>12}   {'source -> target':<28}verb")
        for m, c in counts.most_common():
            pair = kinds[m].most_common(1)[0][0]
            print(f"  {m:<10}{c:>12,}   "
                  f"{pair[0] + ' -> ' + pair[1]:<28}{KNOWN.get(m, '')}")
        print("\nPick with --metaedges, e.g. --metaedges CtD,CpD")
        return

    want = [m.strip() for m in a.metaedges.split(",") if m.strip()]
    names, _kinds = load_nodes(a.nodes)
    print(f"{len(names):,} nodes")

    rows = defaultdict(list)
    missing = Counter()
    for s, m, t in iter_edges(a.edges):
        if m not in want:
            continue
        sn, tn = names.get(s), names.get(t)
        if not sn or not tn:
            # An unjoinable id is dropped rather than passed through as a
            # raw identifier, which would silently become an unmatchable
            # "name" in every downstream measurement.
            missing[m] += 1
            continue
        # Keep the bare identifiers too. mondo_parents.py reads x_id/y_id to
        # learn which ontology terms this knowledge base actually uses, and
        # to take its disease names from here rather than from the ontology's
        # own `name:` field -- the two differ, and the output has to match
        # the target strings in these edge files exactly.
        rows[m].append({"x_id": s.split("::", 1)[-1],
                        "x_name": sn,
                        "y_id": t.split("::", 1)[-1],
                        "y_name": tn,
                        "display_relation": KNOWN.get(m, m)})

    if not rows:
        sys.exit("no edges matched. Run with --inspect to see what is there.")

    os.makedirs(a.outdir, exist_ok=True)
    for m in want:
        r = rows.get(m, [])
        if not r:
            print(f"  {m}: no edges found")
            continue
        path = os.path.join(a.outdir, f"{a.prefix}_{m}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["x_id", "x_name",
                                               "y_id", "y_name",
                                               "display_relation"])
            w.writeheader()
            w.writerows(r)
        uniq_s = len({x["x_name"] for x in r})
        uniq_t = len({x["y_name"] for x in r})
        note = f"   ({missing[m]:,} unjoinable, dropped)" if missing[m] else ""
        print(f"  wrote {path}: {len(r):,} edges, "
              f"{uniq_s:,} drugs, {uniq_t:,} diseases{note}")

    # mondo_parents.py takes a single --primekg file and learns the disease
    # vocabulary from it, so a per-relation file would leave every disease
    # that appears only in the other relation without ancestors. Write the
    # union once and point that flag at it.
    combined = os.path.join(a.outdir, f"{a.prefix}_all.csv")
    allrows = [r for m in want for r in rows.get(m, [])]
    if allrows:
        with open(combined, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["x_id", "x_name", "y_id",
                                               "y_name", "display_relation"])
            w.writeheader()
            w.writerows(allrows)
        print(f"  wrote {combined}: {len(allrows):,} edges "
              f"(union, for --primekg)")

    smallest = min((len(v) for v in rows.values() if v), default=0)
    n = max(50, smallest)
    if smallest and smallest < 600:
        print(f"\nThe smallest relation holds {smallest:,} edges, so --n 600 "
              f"would\nsilently measure fewer. Pass --n {n} and read the "
              f"per-relation counts\nthe tier script prints, not the number "
              f"you asked for.")

    files = " ".join(f"{a.prefix}_{m}.csv" for m in want if rows.get(m))
    print("\nNext, build DOID parents (Hetionet uses the Disease Ontology,\n"
          "not MONDO), then measure:\n"
          "  wget http://purl.obolibrary.org/obo/doid.obo\n"
          f"  python3 mondo_parents.py --obo doid.obo --prefix DOID \\\n"
          f"      --primekg {a.prefix}_all.csv --out doid_child_parent.csv\n"
          f"  python3 name_tiers.py --edges {files} \\\n"
          "      --col-source x_name --col-target y_name "
          "--col-relation display_relation \\\n"
          "      --names redirects --any-names --ancestors doid_child_parent.csv \\\n"
          f"      --ancestor-depth 2 --n {n} --csv het_tiers.csv")


if __name__ == "__main__":
    main()
