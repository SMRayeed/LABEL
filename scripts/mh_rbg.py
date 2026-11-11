#!/usr/bin/env python3
"""Within-drug Mantel-Haenszel odds ratio with two 95% intervals:
the test-based one used in the paper (1/sqrt(sum V)) and the
Robins-Breslow-Greenland (1986) interval, the standard choice.
Standard library only.  usage: mh_rbg.py FILE TIERCOL RELA RELB [label]"""
import csv, math, sys
from collections import defaultdict

def mh(rows, col, ra, rb):
    by = defaultdict(lambda: {ra: [0, 0], rb: [0, 0]})
    for r in rows:
        if r["relation"] in (ra, rb):
            by[r["source"]][r["relation"]][0 if int(r[col]) <= 2 else 1] += 1
    R = S = PR = PSQR = QS = 0.0
    cn = cd = 0.0
    k = 0
    for v in by.values():
        A, B = v[ra]; C, D = v[rb]
        n = A + B + C + D
        if (A + B) == 0 or (C + D) == 0 or n < 2:
            continue
        k += 1
        r_, s_ = A * D / n, B * C / n
        P, Q = (A + D) / n, (B + C) / n
        R += r_; S += s_
        PR += P * r_; PSQR += P * s_ + Q * r_; QS += Q * s_
        cn += A - (A + B) * (A + C) / n
        cd += (A + B) * (C + D) * (A + C) * (B + D) / (n * n * (n - 1))
    orm = R / S
    se_rbg = math.sqrt(PR / (2 * R * R) + PSQR / (2 * R * S) + QS / (2 * S * S))
    se_tb = 1 / math.sqrt(cd)
    chi = (abs(cn) - 0.5) ** 2 / cd
    p = math.erfc(math.sqrt(chi / 2))
    f = lambda se: (math.exp(math.log(orm) - 1.96 * se), math.exp(math.log(orm) + 1.96 * se))
    return k, orm, f(se_tb), f(se_rbg), p

if __name__ == "__main__":
    f, col, ra, rb = sys.argv[1:5]
    lab = sys.argv[5] if len(sys.argv) > 5 else f
    k, orm, tb, rbg, p = mh(list(csv.DictReader(open(f))), col, ra, rb)
    print(f"{lab:28s} drugs {k:4d}  OR {orm:.2f}  test-based [{tb[0]:.2f}, {tb[1]:.2f}]"
          f"  RBG [{rbg[0]:.2f}, {rbg[1]:.2f}]  CMH p {p:.1e}")
