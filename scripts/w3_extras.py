#!/usr/bin/env python3
"""Additional W3PHIAI analyses: retrieval by naming group, and the scale curve.

1. Retrieval helps only when the text names the fact.
   The harder-distractor runs ask every question twice, closed book and with
   the drug's passage. For each relation and naming group (named, family
   only, silent) this reports the change in accuracy the passage brings,
   per model and pooled over the 7-8B models, with 95% bootstrap intervals
   over questions.

2. Scale curve.
   Closed-book probe accuracy for every Qwen2.5 size that has been run, per
   relation, with Wilson intervals, and the indication-minus-contraindication
   gap with a 95% bootstrap interval over drugs.

Writes a markdown report (--out) and the numbers for the figures (--json).
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from name_tiers import wilson  # noqa: E402

GROUPS = [("named", "named"), ("category", "family only"), ("silent", "silent")]


def boot_mean(vals, B=5000, seed=0):
    if not vals:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    n = len(vals)
    m = sum(vals) / n
    bs = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return m, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def tag_of(path):
    return os.path.basename(path).replace("abstain_", "").replace("_clean.csv", "")


def retrieval(files, pool_tags):
    per_q = defaultdict(dict)            # model -> qid -> (rel, group, gain)
    for f in files:
        m = tag_of(f)
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        by = defaultdict(dict)
        for r in rows:
            by[r["id"]][r["condition"]] = r
        for qid, d in by.items():
            if "passage" in d and "closed" in d:
                g = (d["passage"]["outcome"] == "correct") - (d["closed"]["outcome"] == "correct")
                per_q[m][qid] = (d["passage"]["relation"], d["passage"]["group"], g,
                                 d["closed"]["outcome"] == "correct",
                                 d["passage"]["outcome"] == "correct")
    out = {"per_model": {}, "pooled": {}}
    lines = ["## 1. Retrieval helps only when the text names the fact", "",
             "Accuracy closed book and with the drug's passage (%), and the change "
             "(points, 95% bootstrap CI over questions).", ""]
    for m in sorted(per_q):
        lines += [f"**{m}**", "",
                  "| Relation | Group | n | Closed | Passage | Change [95% CI] |",
                  "|---|---|---|---|---|---|"]
        out["per_model"][m] = {}
        for rel in ("indication", "contraindication"):
            for g, gname in GROUPS:
                qs = [v for v in per_q[m].values() if v[0] == rel and v[1] == g]
                if not qs:
                    continue
                c = 100 * sum(v[3] for v in qs) / len(qs)
                p = 100 * sum(v[4] for v in qs) / len(qs)
                mu, lo, hi = boot_mean([100 * v[2] for v in qs])
                out["per_model"][m][f"{rel}|{g}"] = {"n": len(qs), "closed": round(c, 2),
                                                     "passage": round(p, 2),
                                                     "change": [round(mu, 2), round(lo, 2), round(hi, 2)]}
                lines.append(f"| {rel} | {gname} | {len(qs)} | {c:.1f} | {p:.1f} | "
                             f"{mu:+.1f} [{lo:+.1f}, {hi:+.1f}] |")
        lines.append("")
    # pooled over the 7-8B models: average each question's change over models
    pool = [t for t in pool_tags if t in per_q]
    if pool:
        lines += [f"**Pooled over {', '.join(pool)}** (each question averaged over models)", "",
                  "| Relation | Group | n | Change [95% CI] |", "|---|---|---|---|"]
        common = sorted(set.intersection(*(set(per_q[t]) for t in pool)))  # fixed order for the bootstrap
        for rel in ("indication", "contraindication"):
            for g, gname in GROUPS:
                qs = [q for q in common if per_q[pool[0]][q][0] == rel and per_q[pool[0]][q][1] == g]
                vals = [100 * sum(per_q[t][q][2] for t in pool) / len(pool) for q in qs]
                if not vals:
                    continue
                mu, lo, hi = boot_mean(vals)
                out["pooled"][f"{rel}|{g}"] = {"n": len(vals), "change": [round(mu, 2), round(lo, 2), round(hi, 2)]}
                lines.append(f"| {rel} | {gname} | {len(vals)} | {mu:+.1f} [{lo:+.1f}, {hi:+.1f}] |")
        # does the passage help the two relations equally when the text names them?
        qi = [q for q in common if per_q[pool[0]][q][0] == "indication" and per_q[pool[0]][q][1] == "named"]
        qc = [q for q in common if per_q[pool[0]][q][0] == "contraindication" and per_q[pool[0]][q][1] == "named"]
        if qi and qc:
            vi = [100 * sum(per_q[t][q][2] for t in pool) / len(pool) for q in qi]
            vc = [100 * sum(per_q[t][q][2] for t in pool) / len(pool) for q in qc]
            rng = random.Random(1)
            diffs = []
            for _ in range(5000):
                a = [vi[rng.randrange(len(vi))] for _ in vi]
                b = [vc[rng.randrange(len(vc))] for _ in vc]
                diffs.append(sum(a) / len(a) - sum(b) / len(b))
            diffs.sort()
            d0 = sum(vi) / len(vi) - sum(vc) / len(vc)
            out["pooled"]["named_ind_minus_contra"] = [round(d0, 2), round(diffs[124], 2), round(diffs[4874], 2)]
            lines += ["", f"Named group, indication minus contraindication change: "
                          f"{d0:+.1f} points [{diffs[124]:+.1f}, {diffs[4874]:+.1f}]"]
        lines.append("")
    return out, lines


def size_of(tag):
    m = re.search(r"(\d+(?:\.\d+)?)B", tag)
    return float(m.group(1)) if m else float("nan")


def scale(files):
    out = {}
    lines = ["## 2. Scale curve (closed book, five options, chance 20%)", "",
             "| Model | Indication | Off-label use | Contraindication | Ind. minus contra. [95% CI over drugs] |",
             "|---|---|---|---|---|"]
    for f in sorted(files, key=lambda p: size_of(os.path.basename(p))):
        tag = os.path.basename(f).replace("p1_edges_d2_", "").replace(".csv", "")
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        acc = {}
        for rel in ("indication", "off-label use", "contraindication"):
            rr = [r for r in rows if r["relation"] == rel]
            k = sum(int(r["mc_correct"]) for r in rr)
            acc[rel] = [round(x, 2) for x in wilson(k, len(rr))] + [len(rr)]
        # gap with a bootstrap over drugs
        by_drug = defaultdict(lambda: {"indication": [], "contraindication": []})
        for r in rows:
            if r["relation"] in ("indication", "contraindication"):
                by_drug[r["source"]][r["relation"]].append(int(r["mc_correct"]))
        drugs = list(by_drug)
        rng = random.Random(0)
        gaps = []
        for _ in range(2000):
            si = sc = ni = nc = 0
            for _ in drugs:
                d = by_drug[drugs[rng.randrange(len(drugs))]]
                si += sum(d["indication"]); ni += len(d["indication"])
                sc += sum(d["contraindication"]); nc += len(d["contraindication"])
            if ni and nc:
                gaps.append(100 * (si / ni - sc / nc))
        gaps.sort()
        g0 = acc["indication"][0] - acc["contraindication"][0]
        lo, hi = gaps[int(0.025 * len(gaps))], gaps[int(0.975 * len(gaps)) - 1]
        out[tag] = {"params_b": size_of(tag), "acc": acc, "gap": [round(g0, 2), round(lo, 2), round(hi, 2)]}
        lines.append(f"| {tag} | {acc['indication'][0]:.1f} | {acc['off-label use'][0]:.1f} | "
                     f"{acc['contraindication'][0]:.1f} | {g0:+.1f} [{lo:+.1f}, {hi:+.1f}] |")
    lines.append("")
    return out, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--abstain", nargs="*", default=[])
    ap.add_argument("--pool", nargs="*", default=[], help="model tags pooled for section 1")
    ap.add_argument("--probe", nargs="*", default=[], help="p1_edges_d2_<model>.csv files")
    ap.add_argument("--labels-json", default="", help="label_coverage.json, copied into the report")
    ap.add_argument("--out", default="EXTRAS.md")
    ap.add_argument("--json", default="extras.json")
    a = ap.parse_args()

    js, md = {}, ["# W3PHIAI additional analyses", ""]
    files = [f for f in a.abstain if os.path.exists(f)]
    if files:
        js["retrieval"], lines = retrieval(files, a.pool)
        md += lines
    probes = [f for f in a.probe if os.path.exists(f)]
    if probes:
        js["scale"], lines = scale(probes)
        md += lines
    if a.labels_json and os.path.exists(a.labels_json):
        lab = json.load(open(a.labels_json, encoding="utf-8"))
        js["labels"] = lab
        md += ["## 3. FDA labels", "",
               f"Drugs with a label: {lab['drugs_with_label']} of {lab['drugs_total']}.", "",
               "| Relation | n | Wikipedia | Wiki+PubMed | Label, own section | Whole label | All three | Prose misses rescued by label section |",
               "|---|---|---|---|---|---|---|---|"]
        for rel, r in lab["relations"].items():
            md.append(f"| {rel} | {r['n']} | {r['wiki'][0]:.1f} | {r['union'][0]:.1f} | {r['own'][0]:.1f} | "
                      f"{r['any'][0]:.1f} | {r['all3'][0]:.1f} | {r['rescued_by_own_section']:.1f}% |")
        md.append("")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(js, fh, indent=2)
    print("\n".join(md))
    print(f"wrote {a.out} and {a.json}")


if __name__ == "__main__":
    main()
