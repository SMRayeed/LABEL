#!/usr/bin/env python3
"""
Does the indication / contraindication gap survive once you account for how
many edges each drug has?

The objection this answers. PrimeKG carries roughly three times more
contraindication edges than indication edges. A drug with 3 indications and
30 contraindications will name a larger FRACTION of the first set than the
second in a finite article, for purely combinatorial reasons -- no
documentation bias required. The measured ratio of 2.93 sits close enough to
the edge-count ratio of 3.27 that a reviewer will read the second as the
cause of the first.

Matching on article length and page views does not touch this. Those are
properties of the drug; the dilution is a property of the drug-relation pair.

Three tests, strongest first:

  1. WITHIN-DRUG. For drugs carrying both relations, compare the two rates
     for the SAME drug. Every drug-level confound -- article length,
     popularity, how well studied it is -- cancels exactly.
  2. DEGREE-CONTROLLED REGRESSION. Add log(edges for this drug in this
     relation) to the model and see whether the relation term survives it.
  3. DEGREE-MATCHED STRATUM. Keep only drug-relation pairs with comparable
     edge counts and compare raw rates inside that stratum.

If the gap dies in test 1, the finding is dilution and the paper is about
something else. If it survives all three, the objection is answered with
data rather than argument.

    python3 dilution_test.py --scored p1_edges_d2.csv \\
        --edges pkg_ind_d2d.csv pkg_con_d2dis.csv pkg_off_d2dis.csv \\
        --col-source x_name --col-relation display_relation

Standard library only.
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

csv.field_size_limit(min(2**31 - 1, sys.maxsize))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h))


def two_prop(k1, n1, k2, n2):
    if not n1 or not n2:
        return (0.0, 0.0, 0.0, float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se == 0:
        return (100 * p1, 100 * p2, 100 * (p1 - p2), float("nan"), float("nan"))
    z = (p1 - p2) / se
    return (100 * p1, 100 * p2, 100 * (p1 - p2), z,
            math.erfc(abs(z) / math.sqrt(2)))


def signed_rank(diffs):
    """Wilcoxon signed-rank, normal approximation with tie correction."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 6:
        return (float("nan"), float("nan"), n)
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_pos = sum(ranks[i] for i in range(n) if d[i] > 0)
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_pos - mu) / sd if sd else float("nan")
    return (z, math.erfc(abs(z) / math.sqrt(2)), n)


def irls(X, y, names, iters=60, tol=1e-9):
    """Plain logistic regression with Wald standard errors. numpy-free."""
    n, p = len(X), len(X[0])
    b = [0.0] * p
    for _ in range(iters):
        eta = [sum(X[i][j] * b[j] for j in range(p)) for i in range(n)]
        mu = [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, e)))) for e in eta]
        w = [max(m * (1 - m), 1e-9) for m in mu]
        z = [eta[i] + (y[i] - mu[i]) / w[i] for i in range(n)]
        A = [[sum(w[i] * X[i][r] * X[i][c] for i in range(n)) for c in range(p)]
             for r in range(p)]
        rhs = [sum(w[i] * X[i][r] * z[i] for i in range(n)) for r in range(p)]
        M = [A[r][:] + [rhs[r]] for r in range(p)]
        for c in range(p):                      # Gauss-Jordan with pivoting
            piv = max(range(c, p), key=lambda r: abs(M[r][c]))
            if abs(M[piv][c]) < 1e-12:
                return None
            M[c], M[piv] = M[piv], M[c]
            d = M[c][c]
            M[c] = [v / d for v in M[c]]
            for r in range(p):
                if r != c and M[r][c] != 0:
                    f = M[r][c]
                    M[r] = [M[r][k] - f * M[c][k] for k in range(p + 1)]
        nb = [M[r][p] for r in range(p)]
        if max(abs(nb[j] - b[j]) for j in range(p)) < tol:
            b = nb
            break
        b = nb
    eta = [sum(X[i][j] * b[j] for j in range(p)) for i in range(n)]
    mu = [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, e)))) for e in eta]
    w = [max(m * (1 - m), 1e-9) for m in mu]
    A = [[sum(w[i] * X[i][r] * X[i][c] for i in range(n)) for c in range(p)]
         for r in range(p)]
    I = [[1.0 if r == c else 0.0 for c in range(p)] for r in range(p)]
    M = [A[r][:] + I[r][:] for r in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(p):
            if r != c:
                f = M[r][c]
                M[r] = [M[r][k] - f * M[c][k] for k in range(2 * p)]
    out = []
    for j in range(p):
        se = math.sqrt(max(M[j][p + j], 0.0))
        z = b[j] / se if se else float("nan")
        out.append((names[j], b[j], se, z,
                    math.erfc(abs(z) / math.sqrt(2)) if se else float("nan")))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", required=True,
                    help="per-edge file carrying source, relation and tier")
    ap.add_argument("--edges", nargs="+", required=True,
                    help="the FULL PrimeKG relation files, for edge counts")
    ap.add_argument("--col-source", default="x_name")
    ap.add_argument("--col-relation", default="display_relation")
    ap.add_argument("--a", default="indication")
    ap.add_argument("--b", default="contraindication")
    ap.add_argument("--text-cut", type=int, default=2,
                    help="highest tier counted as named (default 2)")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    # --- degree: how many edges each drug carries in each relation ---------
    deg = defaultdict(int)
    rel_total = defaultdict(int)
    for path in a.edges:
        if not os.path.exists(path):
            sys.exit(f"missing edge file: {path}")
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            rd = csv.DictReader(fh)
            if a.col_source not in (rd.fieldnames or []):
                sys.exit(f"{path} has no column {a.col_source}. "
                         f"Columns: {', '.join(rd.fieldnames or [])}")
            for r in rd:
                src = (r.get(a.col_source) or "").strip()
                rel = (r.get(a.col_relation) or "").strip()
                if src and rel:
                    deg[(src, rel)] += 1
                    rel_total[rel] += 1
    print("edges per relation in the full files:")
    for rel, n in sorted(rel_total.items(), key=lambda kv: -kv[1]):
        print(f"  {rel:<22}{n:>9,}")
    if rel_total.get(a.a) and rel_total.get(a.b):
        print(f"\n  edge-count ratio  {a.b} / {a.a}  =  "
              f"{rel_total[a.b] / rel_total[a.a]:.2f}")
        print("  (if the measured coverage ratio lands near this, dilution is "
              "the suspect)\n")

    # --- the measured edges -------------------------------------------------
    rows = []
    with open(a.scored, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                t = int(r["tier"])
            except (KeyError, TypeError, ValueError):
                continue
            src = (r.get("source") or r.get(a.col_source) or "").strip()
            rel = (r.get("relation") or r.get(a.col_relation) or "").strip()
            if not src or rel not in (a.a, a.b):
                continue
            rows.append({"src": src, "rel": rel,
                         "named": 1 if t <= a.text_cut else 0,
                         "deg": deg.get((src, rel), 0)})
    if not rows:
        sys.exit("no rows for those two relations; check --col-source/--a/--b")

    def rate(rel):
        k = sum(r["named"] for r in rows if r["rel"] == rel)
        n = sum(1 for r in rows if r["rel"] == rel)
        return k, n

    ka, na = rate(a.a)
    kb, nb = rate(a.b)
    pa, pb, d, z, pv = two_prop(ka, na, kb, nb)
    print("=" * 74)
    print("BASELINE, no degree control")
    print("=" * 74)
    print(f"  {a.a:<20}{pa:6.2f}%   n={na:,}")
    print(f"  {a.b:<20}{pb:6.2f}%   n={nb:,}")
    print(f"  difference {d:+.2f} pts    ratio {pa / pb if pb else float('nan'):.2f}"
          f"    z={z:.2f}  p={pv:.2e}")

    # --- test 1: within drug, Mantel-Haenszel --------------------------
    # A paired comparison of per-drug RATES is wrong here and reverses the
    # sign. Most drugs contribute one or two indication edges, so that rate
    # quantizes to 0 or 1, while a drug with twenty contraindication edges
    # lands on a smooth fraction above zero. The paired test then counts the
    # quantization, not the effect. Mantel-Haenszel pools 2x2 tables across
    # drugs and weights each by its own size, which is the correct
    # within-drug estimator when the two arms have unequal denominators.
    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # drug -> rel -> [named, not]
    for r in rows:
        cell = by[r["src"]][r["rel"]]
        cell[0 if r["named"] else 1] += 1
    num = den = 0.0
    chi_num = chi_den = 0.0
    strata = 0
    for sname, v in by.items():
        if a.a not in v or a.b not in v:
            continue
        A, B = v[a.a]          # indication: named, not named
        C, D = v[a.b]          # contraindication: named, not named
        n = A + B + C + D
        if n < 2:
            continue
        strata += 1
        num += A * D / n
        den += B * C / n
        e = (A + B) * (A + C) / n
        var = ((A + B) * (C + D) * (A + C) * (B + D) /
               (n * n * (n - 1))) if n > 1 else 0.0
        chi_num += A - e
        chi_den += var
    print("\n" + "=" * 74)
    print("TEST 1  \u2014  WITHIN DRUG, Mantel-Haenszel")
    print("=" * 74)
    if strata < 5 or den <= 0 or chi_den <= 0:
        print(f"  only {strata} drugs carry both relations with usable tables.")
    else:
        or_mh = num / den
        chi = (abs(chi_num) - 0.5) ** 2 / chi_den
        pval = math.erfc(math.sqrt(chi / 2.0))
        se_log = 1.0 / math.sqrt(chi_den) if chi_den > 0 else float("nan")
        lo = math.exp(math.log(or_mh) - 1.96 * se_log)
        hi = math.exp(math.log(or_mh) + 1.96 * se_log)
        print(f"  drugs contributing a table: {strata:,}")
        print(f"  Mantel-Haenszel odds ratio, {a.a} vs {a.b}: "
              f"{or_mh:.2f}   95% CI [{lo:.2f}, {hi:.2f}]")
        print(f"  MH chi-square {chi:.2f}   p = {pval:.2e}")
        print("\n  " + ("the gap holds within drug"
                        if or_mh > 1 and pval < 0.05
                        else "the gap does not hold within drug"))
        print("  This cancels article length, popularity and how well studied")
        print("  the drug is. It does NOT answer the dilution objection, because")
        print("  dilution operates INSIDE a drug: the same article still carries")
        print("  3 indications and 30 contraindications. Only tests 2 and 3,")
        print("  which hold degree fixed, speak to that.")
    print("\n  Note: a paired comparison of per-drug rates is NOT valid here.")
    print("  With one or two indication edges per drug that rate is 0 or 1,")
    print("  so the comparison measures quantization, not coverage.")

    # --- test 2: degree-controlled regression -------------------------------
    print("\n" + "=" * 74)
    print("TEST 2  —  REGRESSION WITH log(edges for this drug in this relation)")
    print("=" * 74)
    use = [r for r in rows if r["deg"] > 0]
    if len(use) < 50:
        print("  too few rows carry a degree; are the edge files the right ones?")
    else:
        X = [[1.0, 1.0 if r["rel"] == a.b else 0.0,
              math.log(r["deg"])] for r in use]
        y = [r["named"] for r in use]
        fit = irls(X, y, ["intercept", f"is_{a.b}", "log_degree"])
        if fit is None:
            print("  the fit did not converge (collinear design)")
        else:
            print(f"  {'term':<20}{'coef':>10}{'se':>9}{'z':>8}{'p':>12}")
            for nm, b_, se, zz, pp in fit:
                print(f"  {nm:<20}{b_:>10.4f}{se:>9.4f}{zz:>8.2f}{pp:>12.2e}")
            rel_term = fit[1]
            print("\n  " + ("the relation term survives the degree control"
                            if rel_term[4] == rel_term[4] and rel_term[4] < 0.05
                            else "the relation term does NOT survive — degree "
                                 "was carrying it"))

    # --- test 3: degree-matched stratum -------------------------------------
    print("\n" + "=" * 74)
    print("TEST 3  —  DEGREE-MATCHED STRATUM")
    print("=" * 74)
    strata = [(1, 2), (3, 5), (6, 12), (13, 10 ** 9)]
    print(f"  {'edges for that drug':<24}{a.a:>12}{a.b:>18}{'diff':>9}{'p':>11}")
    any_row = False
    for lo, hi in strata:
        sel = [r for r in rows if lo <= r["deg"] <= hi]
        k1 = sum(r["named"] for r in sel if r["rel"] == a.a)
        n1 = sum(1 for r in sel if r["rel"] == a.a)
        k2 = sum(r["named"] for r in sel if r["rel"] == a.b)
        n2 = sum(1 for r in sel if r["rel"] == a.b)
        if n1 < 15 or n2 < 15:
            continue
        any_row = True
        p1, p2, dd, zz, pp = two_prop(k1, n1, k2, n2)
        lab = f"{lo} to {hi}" if hi < 10 ** 9 else f"{lo}+"
        print(f"  {lab:<24}{p1:>10.2f}% {f'(n={n1})':<8}"
              f"{p2:>8.2f}% {f'(n={n2})':<8}{dd:>+8.2f}{pp:>11.2e}")
    if not any_row:
        print("  no stratum holds 15 edges of each relation; widen the bins.")
    else:
        print("\n  A gap that holds inside every stratum is not a dilution")
        print("  artifact. A gap that only appears in the wide strata is.")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["src", "rel", "named", "deg"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
