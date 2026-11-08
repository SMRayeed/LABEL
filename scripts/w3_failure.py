#!/usr/bin/env python3
"""
How models fail on the harder-distractor questions.

1. Failure styles, closed book, all questions: share answered wrong, share
   declined ("Not enough information"), and precision (accuracy among
   answered questions), per model and relation.   -> paper, Table "How models fail"
2. Declines by what the page says (named / family only / silent), closed book
   and with the page, averaged over the pooled models.  -> supplement

    python3 w3_failure.py --abstain abstain_*_clean.csv \\
        --pool Qwen2.5-7B-Instruct OLMo-2-1124-7B-Instruct granite-3.1-8b-instruct

Standard library only.
"""
import argparse
import csv
import os

RELS = ("indication", "contraindication")
GROUPS = (("named", "Names the disease"), ("category", "Names only its family"),
          ("silent", "Silent"))


def tag_of(path):
    b = os.path.basename(path)
    return b[len("abstain_"):].rsplit("_clean.csv", 1)[0] if b.startswith("abstain_") else b


def rates(rows):
    n = len(rows)
    if not n:
        return None
    w = sum(r["outcome"] == "wrong" for r in rows)
    d = sum(r["outcome"] == "abstain" for r in rows)
    c = sum(r["outcome"] == "correct" for r in rows)
    return {"n": n, "wrong": 100 * w / n, "decline": 100 * d / n,
            "precision": 100 * c / max(1, n - d)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--abstain", nargs="+", required=True)
    ap.add_argument("--pool", nargs="*", default=[])
    a = ap.parse_args()
    data = {tag_of(f): list(csv.DictReader(open(f, encoding="utf-8"))) for f in a.abstain}

    print("1. Failure styles, closed book, harder distractors (%)\n")
    print(f"{'model':<28}" + "".join(f"{r[:10] + ' ' + k:>18}" for r in RELS
                                      for k in ("wrong", "decl.", "prec.")))
    for m, rows in data.items():
        line = f"{m:<28}"
        for rel in RELS:
            x = rates([r for r in rows if r["condition"] == "closed" and r["relation"] == rel])
            line += "".join(f"{x[k]:>18.1f}" for k in ("wrong", "decline", "precision"))
        print(line)

    pool = [m for m in a.pool if m in data]
    if pool:
        print(f"\n2. Share declining (%), averaged over {', '.join(pool)}\n")
        print(f"{'page':<24}" + "".join(f"{r[:10] + ' ' + c:>20}" for r in RELS
                                        for c in ("closed", "page")))
        for g, gname in GROUPS:
            line = f"{gname:<24}"
            for rel in RELS:
                for cond in ("closed", "passage"):
                    vals = [rates([r for r in data[m] if r["condition"] == cond and
                                   r["relation"] == rel and r["group"] == g])["decline"]
                            for m in pool]
                    line += f"{sum(vals) / len(vals):>20.1f}"
            print(line)


if __name__ == "__main__":
    main()
