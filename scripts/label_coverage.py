#!/usr/bin/env python3
"""How often the FDA label names each fact, measured like the prose corpora.

Every edge of the drug-matched sample is scored with the same matcher as
Wikipedia and PubMed (exact name, Wikipedia redirect names, MONDO ancestors
up to --depth levels; word boundaries; phrases under four characters
skipped). Tiers: 0 exact, 1-2 alternative name, 3 family only, 4 nothing.

Three label texts are scored:
  own   the section that records the relation: Indications and Usage for an
        indication, Contraindications for a contraindication
  any   the whole label
Only drugs with a label are kept, and the prose figures below are restricted
to the same drugs so the corpora are compared on the same edges.

Outputs (in the working directory):
  me_label_own.csv, me_label_any.csv   matched_edges.csv rows with 'tier'
                                       replaced, for dilution_test.py
  me_label_allthree.csv                best tier over Wikipedia, PubMed and
                                       the whole label
  --json                               numbers for the paper figure
"""
import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from name_tiers import (climb, hit_tiers, load_ancestors, read_names,  # noqa: E402
                        tier_sets, wilson)
from fetch_labels import safe  # noqa: E402

EMPTY = {"synonyms": [], "vernacular": []}


def tier_in(text, sets):
    if not text:
        return 4
    hits = hit_tiers(text, sets)
    for i, h in enumerate(hits):
        if h:
            return i if i < 3 else 3
    return 4


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges", default="matched_edges.csv")
    ap.add_argument("--union", default="me_bal_union.csv",
                    help="Wikipedia + PubMed tiers for the same edges")
    ap.add_argument("--labels", default="label_cache")
    ap.add_argument("--names-cache", default="gbif_cache")
    ap.add_argument("--ancestors", default="mondo_child_parent.csv")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--floor-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    anc = load_ancestors(a.ancestors)
    rows = list(csv.DictReader(open(a.edges, encoding="utf-8")))
    union = {r["id"]: r for r in csv.DictReader(open(a.union, encoding="utf-8"))}

    labels = {}
    for d in sorted({r["source"] for r in rows}):
        p = os.path.join(a.labels, safe(d) + ".json")
        if os.path.exists(p):
            rec = json.load(open(p, encoding="utf-8"))
            if rec.get("found"):
                labels[d] = rec
    n_drugs = len({r["source"] for r in rows})
    print(f"drugs with an FDA label: {len(labels):,} of {n_drugs:,}")
    if not labels:
        sys.exit("no labels in the cache; run fetch_labels.py first")

    def sets_for(target):
        n = dict(read_names(a.names_cache, "redir", target) or EMPTY)
        n["synonyms"] = []
        n["ancestors"] = climb(target, anc, a.depth)
        return tier_sets(target, n)

    # chance floor: exact names of unrelated diseases from the same relation
    rng = random.Random(a.seed)
    partners = {}
    targets_by_rel = {}
    for r in rows:
        partners.setdefault(r["source"], set()).add(r["target"].lower())
        targets_by_rel.setdefault(r["relation"], set()).add(r["target"])
    targets_by_rel = {k: sorted(v) for k, v in targets_by_rel.items()}

    out_own, out_any, out_all3 = [], [], []
    per = {}   # relation -> counters
    for r in rows:
        lab = labels.get(r["source"])
        if lab is None:
            continue
        rel = r["relation"]
        sets = sets_for(r["target"])
        own_text = lab["indications"] if rel == "indication" else (
            lab["contraindications"] if rel == "contraindication" else lab["full"])
        t_own = tier_in(own_text, sets)
        t_any = tier_in(lab["full"], sets)
        u = union.get(r["id"], {})
        t_wiki = int(r["tier"])
        t_union = int(u.get("tier_union", u.get("tier", 4)) or 4)
        t_all3 = min(t_union, t_any)
        for lst, t in ((out_own, t_own), (out_any, t_any), (out_all3, t_all3)):
            rr = dict(r)
            rr["tier"] = t
            lst.append(rr)
        # floor on the whole label
        pool = [t for t in targets_by_rel[rel] if t.lower() not in partners[r["source"]]]
        fl = sum(tier_in(lab["full"], [[x], [x], [x], [x]]) == 0
                 for x in rng.sample(pool, min(a.floor_k, len(pool))))
        c = per.setdefault(rel, {"n": 0, "wiki": 0, "union": 0, "own": 0, "any": 0,
                                 "all3": 0, "own_fam": 0, "any_fam": 0,
                                 "prose_miss": 0, "rescued_own": 0, "rescued_any": 0,
                                 "floor_hits": 0, "floor_n": 0, "own_empty": 0})
        c["n"] += 1
        c["wiki"] += t_wiki <= 2
        c["union"] += t_union <= 2
        c["own"] += t_own <= 2
        c["any"] += t_any <= 2
        c["all3"] += t_all3 <= 2
        c["own_fam"] += t_own == 3
        c["any_fam"] += t_any == 3
        c["own_empty"] += not own_text.strip()
        if t_union > 2:
            c["prose_miss"] += 1
            c["rescued_own"] += t_own <= 2
            c["rescued_any"] += t_any <= 2
        c["floor_hits"] += fl
        c["floor_n"] += min(a.floor_k, len(pool))

    for name, lst in (("me_label_own.csv", out_own), ("me_label_any.csv", out_any),
                      ("me_label_allthree.csv", out_all3)):
        with open(name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(lst[0].keys()))
            w.writeheader()
            w.writerows(lst)
        print(f"wrote {name}: {len(lst):,} rows")

    def pct(k, n):
        p, lo, hi = wilson(k, n)
        return p, lo, hi

    print("\nShare of facts named (tier 0-2), same drugs, same edges (% [95% Wilson CI])")
    print(f"{'relation':<18}{'n':>6}  {'Wikipedia':>20}{'Wiki+PubMed':>20}"
          f"{'label, own section':>22}{'label, whole':>20}{'all three':>20}")
    summary = {"drugs_with_label": len(labels), "drugs_total": n_drugs, "relations": {}}
    for rel in sorted(per):
        c = per[rel]
        cells, js = [], {"n": c["n"]}
        for k in ("wiki", "union", "own", "any", "all3"):
            p, lo, hi = pct(c[k], c["n"])
            cells.append(f"{p:6.2f} [{lo:5.1f},{hi:5.1f}]")
            js[k] = [round(p, 2), round(lo, 2), round(hi, 2)]
        js["own_family_only"] = round(100 * c["own_fam"] / c["n"], 2)
        js["any_family_only"] = round(100 * c["any_fam"] / c["n"], 2)
        js["own_section_empty"] = round(100 * c["own_empty"] / c["n"], 2)
        js["prose_miss"] = c["prose_miss"]
        js["rescued_by_own_section"] = round(100 * c["rescued_own"] / max(1, c["prose_miss"]), 2)
        js["rescued_by_whole_label"] = round(100 * c["rescued_any"] / max(1, c["prose_miss"]), 2)
        js["floor_exact_whole_label"] = round(100 * c["floor_hits"] / max(1, c["floor_n"]), 2)
        summary["relations"][rel] = js
        print(f"{rel:<18}{c['n']:>6}  " + "  ".join(f"{x:>18}" for x in cells))
    print("\nFacts the prose misses (Wikipedia + PubMed, tier > 2) that the label names:")
    for rel, js in summary["relations"].items():
        print(f"  {rel:<18} {js['prose_miss']:>5} missed; named in own section "
              f"{js['rescued_by_own_section']:.1f}%, anywhere in label "
              f"{js['rescued_by_whole_label']:.1f}%")
    print("\nFamily only (tier 3) in the label's own section:")
    for rel, js in summary["relations"].items():
        print(f"  {rel:<18} {js['own_family_only']:.2f}%   (own section empty for "
              f"{js['own_section_empty']:.1f}% of edges)")
    print("\nChance floor: exact name of an unrelated disease anywhere in the label")
    for rel, js in summary["relations"].items():
        print(f"  {rel:<18} {js['floor_exact_whole_label']:.2f}%")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
