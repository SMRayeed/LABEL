#!/usr/bin/env python3
"""Routing test: does sending safety questions to the FDA label help?

Reads, for each model, the usual harder-option run (abstain_<tag>_clean.csv,
closed book and the prose page) and the same questions with the passage cut
from the drug's whole FDA label (abstain_<tag>_label.csv, from
abstain_probe.py --page-labels). Only questions whose drug has a label are
compared, so every condition answers the same questions.

Four conditions per question:
  closed   no passage
  prose    Wikipedia + PubMed page (what a retriever returns today)
  label    whole-label page for every question
  routed   indications read prose, contraindications read the label
           (the relation-aware design of the paper's router figure)

Outcomes are correct, wrong and decline (Not enough information), in percent.
Pooled rows average each question over the models first. The interval is a
bootstrap over questions (5,000 draws, seed 0) for routed minus prose.

    python3 route_report.py --models Qwen2.5-7B-Instruct OLMo-2-1124-7B-Instruct \\
        granite-3.1-8b-instruct --dir .
"""
import argparse
import csv
import os
import random
import sys
from collections import defaultdict

RELS = ("indication", "contraindication")
OUTS = ("correct", "wrong", "abstain")


def read(path, cond_map):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        c = cond_map.get(r["condition"])
        if c:
            out[(r["id"], c)] = (r["relation"], r["group"], r["outcome"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="model tags")
    ap.add_argument("--dir", default=".")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--label-tiers", default="me_label_any.csv",
                    help="whole-label naming tiers, to split by whether the label names the answer")
    a = ap.parse_args()

    per = {}          # tag -> {(id, cond): (rel, group, outcome)}
    for t in a.models:
        base = os.path.join(a.dir, f"abstain_{t}_clean.csv")
        lab = os.path.join(a.dir, f"abstain_{t}_label.csv")
        if not (os.path.exists(base) and os.path.exists(lab)):
            print(f"skip {t}: need {base} and {lab}", file=sys.stderr)
            continue
        d = read(base, {"closed": "closed", "passage": "prose"})
        d.update(read(lab, {"passage": "label"}))
        per[t] = d
    if not per:
        sys.exit("no model has both runs")

    def outcome(d, qid, rel, cond):
        if cond == "routed":
            cond = "prose" if rel == "indication" else "label"
        v = d.get((qid, cond))
        return v[2] if v else None

    conds = ("closed", "prose", "label", "routed")
    tags = list(per)
    qs = sorted({k[0] for d in per.values() for k in d if k[1] == "label"})
    rel_of = {}
    for d in per.values():
        for (qid, _c), (rel, _g, _o) in d.items():
            rel_of[qid] = rel
    # keep questions every model answered in all three runs
    qs = [q for q in qs if all(all((q, c) in per[t] for c in ("closed", "prose", "label"))
                               for t in tags)]
    print(f"{len(qs):,} questions with a label page, answered by all of: {', '.join(tags)}")

    def shares(qset, cond, models):
        acc = {o: 0.0 for o in OUTS}
        for q in qset:
            for o in OUTS:
                acc[o] += sum(outcome(per[t], q, rel_of[q], cond) == o
                              for t in models) / len(models)
        n = max(1, len(qset))
        return {o: 100 * acc[o] / n for o in OUTS}

    for label, models in [(t, [t]) for t in tags] + [("pooled", tags)]:
        print(f"\n=== {label}")
        print(f"  {'relation':<17} {'condition':<8} {'n':>5}  correct  wrong  decline")
        for rel in RELS + ("all",):
            qset = [q for q in qs if rel == "all" or rel_of[q] == rel]
            for c in conds:
                if rel != "all" and c == "routed":
                    continue        # routed equals prose or label within one relation
                s = shares(qset, c, models)
                print(f"  {rel:<17} {c:<8} {len(qset):5d}  {s['correct']:7.1f} "
                      f"{s['wrong']:6.1f} {s['abstain']:8.1f}")

    # split by whether the whole label names the answer (tier <= 2)
    if a.label_tiers and os.path.exists(os.path.join(a.dir, a.label_tiers)):
        named = {}
        for r in csv.DictReader(open(os.path.join(a.dir, a.label_tiers), encoding="utf-8")):
            named[(r["relation"], r["source"], r["target"])] = int(r["tier"]) <= 2
        def lab_named(q):
            p = q.split("|", 3)
            return named.get((p[0], p[2], p[3])) if len(p) == 4 else None
        print("\n=== pooled, split by whether the whole label names the answer")
        print(f"  {'relation':<17} {'label names it':<15} {'n':>4}  cond   correct  wrong  decline")
        for rel in RELS:
            for flag in (True, False):
                qset = [q for q in qs if rel_of[q] == rel and lab_named(q) is flag]
                if not qset:
                    continue
                for c in ("prose", "label"):
                    sh = shares(qset, c, tags)
                    print(f"  {rel:<17} {'yes' if flag else 'no':<15} {len(qset):4d}  {c:<6}"
                          f" {sh['correct']:6.1f} {sh['wrong']:6.1f} {sh['abstain']:8.1f}")
                # label minus prose with a bootstrap interval over questions
                rng = random.Random(0)
                for o in OUTS:
                    d = [100 * sum((outcome(per[t], q, rel_of[q], "label") == o) -
                                   (outcome(per[t], q, rel_of[q], "prose") == o)
                                   for t in tags) / len(tags) for q in qset]
                    n = len(d)
                    draws = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n
                                   for _ in range(a.boot))
                    lo, hi = draws[int(0.025 * a.boot)], draws[int(0.975 * a.boot) - 1]
                    print(f"  {'':<17} {'':<15} {'':>4}  diff   {o:<8} {sum(d) / n:+6.1f}"
                          f" [{lo:+.1f}, {hi:+.1f}]")

    # bootstrap: label minus prose per relation, routed minus prose overall.
    # Per-question pooled differences are computed once, then resampled.
    rng = random.Random(0)
    print("\n=== pooled differences, 95% bootstrap CI over questions")
    for rel, c1, c2 in (("contraindication", "label", "prose"),
                        ("indication", "label", "prose"),
                        ("all", "routed", "prose")):
        qset = [q for q in qs if rel == "all" or rel_of[q] == rel]
        for o in OUTS:
            d = [100 * sum((outcome(per[t], q, rel_of[q], c1) == o) -
                           (outcome(per[t], q, rel_of[q], c2) == o) for t in tags) / len(tags)
                 for q in qset]
            n = len(d)
            est = sum(d) / n
            draws = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n
                           for _ in range(a.boot))
            lo, hi = draws[int(0.025 * a.boot)], draws[int(0.975 * a.boot) - 1]
            print(f"  {rel:<17} {c1} - {c2}  {o:<8} {est:+6.1f} [{lo:+.1f}, {hi:+.1f}]")

if __name__ == "__main__":
    main()
