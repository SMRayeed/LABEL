#!/usr/bin/env python3
"""
Build a DIRECTED child -> parent disease table for name_tiers.py --ancestors.

PrimeKG stores disease_disease parent-child edges in both directions, so which
end is the parent cannot be recovered from PrimeKG alone. The source ontology
does carry the direction: each term's `is_a:` lines name its parents and
nothing else.

This reads an OBO ontology, keeps only is_a edges, maps ontology ids to the
disease names the knowledge base uses (taken from its own id/name columns so
the strings match exactly), and writes child,parent.

Works for PrimeKG with MONDO (the default) and for Hetionet with the Disease
Ontology via --prefix DOID and doid.obo:

    wget https://purl.obolibrary.org/obo/doid.obo
    python3 mondo_parents.py --obo doid.obo --primekg <hetionet-disease-file> \
        --prefix DOID --out doid_child_parent.csv

It also reports how broad each parent is -- how many descendants it has -- and
drops the broadest, because a term like "disease" or "disorder" appears in
nearly every clinical article and would inflate the granularity tier instead
of measuring it. That cut is the reason this script exists rather than a
one-line awk.

    # 1. get MONDO (about 100 MB; it is a plain text ontology)
    wget https://purl.obolibrary.org/obo/mondo.obo
    #    if that redirect is blocked:
    # wget https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo

    # 2. build the table
    python3 mondo_parents.py --obo mondo.obo --primekg pkg_pc.csv \\
        --out mondo_child_parent.csv

    # 3. measure the granularity tier with it
    python3 name_tiers.py --edges pkg_ind_d2d.csv pkg_con_d2dis.csv \\
        pkg_off_d2dis.csv --names redirects --any-names --n 600 \\
        --ancestors mondo_child_parent.csv --ancestor-depth 1 \\
        --rps 3 --csv primekg_tiers.csv

Stdlib only.
"""

import argparse
import csv
import gzip
import os
import re
import sys
from collections import defaultdict, deque

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

NAME_RE = re.compile(r"^name:\s*(.+?)\s*$")
OBSOLETE_RE = re.compile(r"^is_obsolete:\s*true", re.I)

# The id prefix and zero-padding differ by ontology: MONDO:0002816 is seven
# digits, DOID:1485 is unpadded. Both are set from --prefix / --pad so the
# same script serves PrimeKG (MONDO) and Hetionet (Disease Ontology).
PREFIXES = {
    "MONDO": 7,     # PrimeKG diseases
    "DOID": 0,      # Hetionet diseases, Disease Ontology
    "HP": 7,        # phenotypes, if ever needed
    "EFO": 7,
}


def parse_obo(path, prefix="MONDO"):
    """term id -> {"name": str, "parents": [id, ...]} for non-obsolete terms."""
    id_re = re.compile(rf"^id:\s*({prefix}:\d+)", re.I)
    isa_re = re.compile(rf"^is_a:\s*({prefix}:\d+)", re.I)
    opener = gzip.open if path.lower().endswith(".gz") else open
    terms = {}
    cur = None
    in_term = False
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("["):
                if cur and cur.get("id") and not cur.get("obsolete"):
                    terms[cur["id"]] = {"name": cur.get("name", ""),
                                        "parents": cur.get("parents", [])}
                in_term = line.strip() == "[Term]"
                cur = {"parents": []} if in_term else None
                continue
            if not in_term or cur is None:
                continue
            m = id_re.match(line)
            if m:
                cur["id"] = m.group(1)
                continue
            m = NAME_RE.match(line)
            if m and "name" not in cur:
                cur["name"] = m.group(1)
                continue
            m = isa_re.match(line)
            if m:
                cur["parents"].append(m.group(1))
                continue
            if OBSOLETE_RE.match(line):
                cur["obsolete"] = True
    if cur and cur.get("id") and not cur.get("obsolete"):
        terms[cur["id"]] = {"name": cur.get("name", ""),
                            "parents": cur.get("parents", [])}
    return terms


def primekg_id_names(path):
    """MONDO numeric id -> the disease name PrimeKG prints for it.

    PrimeKG's x_id for a disease is the MONDO id without the prefix and
    without zero padding, so '2816' here is MONDO:0002816. Taking the names
    from PrimeKG rather than from MONDO's own `name:` field matters: the
    output has to match the target strings in the edge files exactly, and the
    two vocabularies differ in places ('anemia (disease)' is PrimeKG's)."""
    cols = None
    out = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames or []
        need = [("x_id", "x_name"), ("y_id", "y_name")]
        if not all(c in cols for pair in need for c in pair):
            sys.exit(f"{path} needs x_id, x_name, y_id, y_name.\n"
                     f"Columns are: {', '.join(cols)}")
        for row in rd:
            for c_id, c_name in need:
                raw = (row.get(c_id) or "").strip()
                nm = (row.get(c_name) or "").strip()
                if raw and nm:
                    out.setdefault(raw, nm)
    return out


def term_key(raw, prefix="MONDO", pad=7):
    """A knowledge base's bare id -> the ontology's full id.

    PrimeKG stores '2816' for MONDO:0002816, so it needs zero-padding to
    seven digits. Hetionet stores 'DOID:1485' or '1485' for Disease Ontology,
    which is unpadded. Both shapes are accepted."""
    raw = str(raw).strip()
    if raw.upper().startswith(prefix.upper() + ":"):
        raw = raw.split(":", 1)[1]
    try:
        n = int(raw)
    except ValueError:
        return None
    return f"{prefix}:{n:0{pad}d}" if pad else f"{prefix}:{n}"


def descendant_counts(child_to_parents):
    """How many terms sit under each term, for the breadth cut."""
    kids = defaultdict(list)
    for c, ps in child_to_parents.items():
        for p in ps:
            kids[p].append(c)
    counts = {}
    for node in kids:
        seen, q = set(), deque(kids[node])
        while q:
            n = q.popleft()
            if n in seen:
                continue
            seen.add(n)
            q.extend(kids.get(n, ()))
        counts[node] = len(seen)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obo", required=True, help="mondo.obo (or .obo.gz)")
    ap.add_argument("--primekg", required=True, dest="kb",
                    help="a knowledge-base file with x_id/x_name/y_id/y_name, "
                         "such as PrimeKG's pkg_pc.csv")
    ap.add_argument("--prefix", default="MONDO", choices=sorted(PREFIXES),
                    help="ontology id prefix: MONDO for PrimeKG, DOID for "
                         "Hetionet (default MONDO)")
    ap.add_argument("--out", default="mondo_child_parent.csv")
    ap.add_argument("--max-descendants", type=int, default=400,
                    help="drop parents with more descendants than this; they "
                         "are terms too broad to be evidence of anything "
                         "(default 400)")
    ap.add_argument("--keep-broad", action="store_true",
                    help="keep every parent, however broad; for checking what "
                         "the cut removes")
    a = ap.parse_args()

    for p in (a.obo, a.kb):
        if not os.path.exists(p):
            sys.exit(f"no such file: {p}")

    print(f"reading {a.obo} ...", file=sys.stderr)
    pad = PREFIXES[a.prefix]
    terms = parse_obo(a.obo, a.prefix)
    child_to_parents = {t: v["parents"] for t, v in terms.items() if v["parents"]}
    print(f"  {len(terms):,} terms, {len(child_to_parents):,} with a parent",
          file=sys.stderr)
    if not child_to_parents:
        sys.exit(f"no {a.prefix} is_a edges found. Wrong --prefix for this "
                 f"file, or not an OBO file?")

    # direction sanity: a real hierarchy is almost never symmetric
    pairs = {(c, p) for c, ps in child_to_parents.items() for p in ps}
    rev = sum(1 for c, p in pairs if (p, c) in pairs)
    print(f"  {rev:,} of {len(pairs):,} is_a pairs appear reversed "
          f"({'directed, good' if rev / len(pairs) < 0.01 else 'NOT DIRECTED'})",
          file=sys.stderr)

    print(f"reading names from {a.kb} ...", file=sys.stderr)
    id_names = primekg_id_names(a.kb)
    by_mondo = {}
    for raw, nm in id_names.items():
        k = term_key(raw, a.prefix, pad)
        if k:
            by_mondo.setdefault(k, nm)
    print(f"  {len(by_mondo):,} {a.prefix} ids carry a name in that file",
          file=sys.stderr)
    overlap = len(set(by_mondo) & set(terms))
    rate = overlap / max(1, len(by_mondo))
    print(f"  {overlap:,} of them exist in this {a.prefix} release "
          f"({100 * rate:.1f}%)", file=sys.stderr)
    # The test is the match RATE, not the count. A small knowledge base can be
    # entirely healthy: Hetionet's drug-disease layer carries 91 DOID ids and
    # matches 86 of them. An absolute floor would reject that while passing a
    # large file matching 3% of its ids, which is the failure actually worth
    # catching -- the wrong ontology, or ids that are not this prefix at all.
    if overlap < 10 or rate < 0.30:
        sys.exit(f"only {100 * rate:.1f}% of ids match ({overlap:,} of "
                 f"{len(by_mondo):,}).\nThat file's ids may not be "
                 f"{a.prefix}; check its x_source column and try another\n"
                 f"--prefix. If they are {a.prefix}, the knowledge base and "
                 f"this ontology release\nmay have drifted apart, and an "
                 f"archived release matching the base is the fix.")
    if rate < 0.80:
        print(f"  note: {len(by_mondo) - overlap:,} ids are absent from this "
              f"release.\n  Those edges get no ancestor tier, which lowers "
              f"tier 3 coverage.", file=sys.stderr)

    counts = descendant_counts(child_to_parents) if not a.keep_broad else {}

    rows, dropped_broad, dropped_unnamed = [], 0, 0
    for child, parents in child_to_parents.items():
        c_name = by_mondo.get(child)
        if not c_name:
            continue
        for parent in parents:
            p_name = by_mondo.get(parent) or (terms.get(parent) or {}).get("name")
            if not p_name:
                dropped_unnamed += 1
                continue
            if not a.keep_broad and counts.get(parent, 0) > a.max_descendants:
                dropped_broad += 1
                continue
            if c_name.strip().lower() == p_name.strip().lower():
                continue
            rows.append((c_name, p_name))

    rows = sorted(set(rows))
    if not rows:
        sys.exit("no usable child-parent pairs survived; try --keep-broad to "
                 "see what the cut removed")

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["child", "parent"])
        w.writerows(rows)

    kids = defaultdict(int)
    for c, _p in rows:
        kids[c] += 1
    print(f"\nwrote {a.out}: {len(rows):,} pairs, {len(kids):,} children, "
          f"{max(kids.values())} parents at most", file=sys.stderr)
    print(f"  dropped {dropped_broad:,} pairs whose parent has more than "
          f"{a.max_descendants:,} descendants", file=sys.stderr)
    print(f"  dropped {dropped_unnamed:,} pairs whose parent has no name",
          file=sys.stderr)

    if not a.keep_broad and counts:
        widest = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
        print("\nbroadest terms, excluded by the cut:", file=sys.stderr)
        for t, n in widest:
            nm = by_mondo.get(t) or (terms.get(t) or {}).get("name", t)
            print(f"  {n:>7,} descendants  {nm}", file=sys.stderr)

    print("\nsample pairs:", file=sys.stderr)
    for c, p in rows[:8]:
        print(f"  {c}  ->  {p}", file=sys.stderr)
    print(f"""
Next, measure the granularity tier against it. Start at depth 1, which asks
only whether the article named the disease's immediate parent:

  python3 name_tiers.py --edges pkg_ind_d2d.csv pkg_con_d2dis.csv \\
      pkg_off_d2dis.csv --names redirects --any-names --n 600 \\
      --ancestors {a.out} --ancestor-depth 1 --rps 3 --csv primekg_tiers.csv

Watch the T3 floor. Under about 15% the column is real. If it climbs, the
parents still in the table are too broad: lower --max-descendants here and
rebuild.""", file=sys.stderr)


if __name__ == "__main__":
    main()
