#!/usr/bin/env python3
"""Does the sentence that names the disease also state the relation?

The text measurement counts a fact as named when the drug's text contains the
disease's exact or alternative name. A reviewer can ask whether that sentence
says the drug treats the disease, must not be given with it, or only mentions
it. This script answers that on the same edges and the same texts.

For every indication and contraindication edge, and for each corpus
(Wikipedia, PubMed, the whole FDA label), it finds the sentences that name the
disease and checks them in two ways:

  cue     relation words in the same sentence. Indication: treat, indicated,
          therapy, used for ... Contraindication: contraindicated, must not,
          should not, avoid, do not use ... plus a looser warning set
          (caution, warning, worsen, exacerbate, risk).
  judge   (--judge MODEL) a model reads the sentence and picks one of four
          letters: the drug treats the disease / must not be used with it /
          may cause, worsen or need caution with it / no link stated.

It then reports, per corpus, the share of facts named, named with a matching
cue and named with the judge's support, and the within-drug Mantel-Haenszel
odds ratio of indication against contraindication for each definition, with
Robins-Breslow-Greenland 95% intervals.

It also writes two sheets for a person to fill in:

  --audit N    N named sentences per relation for Wikipedia and for the label,
               shuffled, with the relation hidden (audit_sheet.csv), and the
               answer key (audit_key.csv). Score the filled sheet with
               --score-audit audit_sheet.csv.
  --recall N   N contraindications whose label has a non-empty
               Contraindications section that the matcher finds nothing in
               (recall_sheet.csv), to read whether the fact is there in other
               words.

    python3 relcheck.py --edges matched_edges.csv --wiki-cache wiki_cache \\
        --pubmed-cache pubmed_cache --labels label_cache \\
        --names-cache gbif_cache --ancestors mondo_child_parent.csv \\
        --audit 40 --recall 50 --out relcheck_sentences.csv

    # the same, with a model judge (CPU, about an hour for a 7B model)
    python3 relcheck.py ... --judge Qwen/Qwen2.5-7B-Instruct --threads 12

    # after filling in the sheets (one *_label column per annotator)
    python3 relcheck.py --score-audit audit_sheet.csv --key audit_key.csv \
        --score-recall recall_sheet.csv
"""
import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import name_tiers as nt  # noqa: E402
from fetch_labels import safe  # noqa: E402

csv.field_size_limit(min(2**31 - 1, sys.maxsize))
RELS = ("indication", "contraindication")
CORPORA = ("wikipedia", "pubmed", "label")

CUE_TREAT = re.compile(
    r"\b(treat\w*|indicated|indication\w*|therap\w*|used (?:for|in|to)|"
    r"management of|manage\w*|approved for|first[- ]line|effective (?:in|for|against)|"
    r"relie\w*|prevent\w*|prophyla\w*|cure\w*)\b", re.I)
CUE_CONTRA = re.compile(
    r"\b(contraindicat\w*|must not|should not|not be (?:used|given|taken|administered)|"
    r"do not (?:use|give|take|administer)|avoid\w*|not recommended|"
    r"should be avoided|is not to be)\b", re.I)
CUE_WARN = re.compile(
    r"\b(caution\w*|warning\w*|worsen\w*|exacerbat\w*|aggravat\w*|risk\w*|"
    r"precaution\w*|adverse|toxic\w*|induced?|may cause|can cause)\b", re.I)

JUDGE_OPTS = [
    "{d} is used to treat or manage {x}.",
    "{d} must not be used in patients with {x}.",
    "{d} may cause or worsen {x}, or needs caution in patients with {x}.",
    "The sentence mentions both but states no link between them.",
]
JUDGE_CODE = ["treat", "contra", "warn", "none"]

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n+")


# ------------------------------------------------------------------- text

def sentences(text):
    out = []
    for s in SENT_SPLIT.split(text or ""):
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) >= 20:
            out.append(s)
    return out


def clean(text):
    try:
        from recover import clean_wikitext
        return clean_wikitext(text)
    except Exception:                                    # noqa: BLE001
        return text


def load_texts(a, drug, cache):
    if drug in cache:
        return cache[drug]
    t = {}
    w = nt.read_article(a.wiki_cache, drug)
    t["wikipedia"] = clean(w) if w else None
    # the balanced two-query corpus of the paper (second_corpus.py suffix)
    p = os.path.join(a.pubmed_cache, f"{nt._safe(drug)}{a.pubmed_suffix}.txt")
    t["pubmed"] = None
    if os.path.exists(p):
        with open(p, encoding="utf-8", errors="replace") as fh:
            t["pubmed"] = fh.read()
    t["label"] = None
    t["label_contra"] = None
    p = os.path.join(a.labels, safe(drug) + ".json")
    if os.path.exists(p):
        rec = json.load(open(p, encoding="utf-8"))
        if rec.get("found"):
            t["label"] = rec.get("full") or ""
            t["label_contra"] = rec.get("contraindications") or ""
    cache[drug] = t
    return t


def matched_sentences(text, phrases, keep=3):
    if not text:
        return []
    out = []
    for s in sentences(text):
        if any(nt.mentions(s, p) for p in phrases):
            out.append(s)
            if len(out) >= keep:
                break
    return out


# -------------------------------------------------------------- statistics

def mh_rbg(cells):
    """cells: drug -> {rel: [yes, no]}. Within-drug MH odds ratio of
    indication against contraindication, RBG 95% CI, CMH p-value."""
    R = S = PR = PSQR = QS = cn = cd = 0.0
    for v in cells.values():
        A, B = v["indication"]
        C, D = v["contraindication"]
        n = A + B + C + D
        if A + B == 0 or C + D == 0 or n < 2:
            continue
        r_, s_ = A * D / n, B * C / n
        P, Q = (A + D) / n, (B + C) / n
        R += r_; S += s_
        PR += P * r_; PSQR += P * s_ + Q * r_; QS += Q * s_
        cn += A - (A + B) * (A + C) / n
        cd += (A + B) * (C + D) * (A + C) * (B + D) / (n * n * (n - 1))
    if R <= 0 or S <= 0 or cd <= 0:
        return float("nan"), (float("nan"), float("nan")), float("nan")
    orm = R / S
    se = math.sqrt(PR / (2 * R * R) + PSQR / (2 * R * S) + QS / (2 * S * S))
    chi = (abs(cn) - 0.5) ** 2 / cd
    return (orm, (math.exp(math.log(orm) - 1.96 * se),
                  math.exp(math.log(orm) + 1.96 * se)),
            math.erfc(math.sqrt(chi / 2)))


def kappa(pairs):
    pairs = [(x, y) for x, y in pairs if x and y]
    if not pairs:
        return float("nan"), 0
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    cx, cy = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(cx[k] * cy[k] for k in set(cx) | set(cy)) / (n * n)
    return ((po - pe) / (1 - pe) if pe < 1 else 1.0), n


# ------------------------------------------------------------------ judge

def judge_rows(a, rows):
    from abstain_probe import LetterScorer, LETTERS
    sc = LetterScorer(a.judge, threads=a.threads, batch_size=a.batch_size)
    prompts = []
    for r in rows:
        d, x = r["drug"], r["disease"]
        opts = "\n".join(f"{LETTERS[i]}. {o.format(d=d, x=x)}"
                         for i, o in enumerate(JUDGE_OPTS))
        prompts.append(f"Sentence: {r['sentence']}\n\n"
                       f"What does this sentence say about {d} and {x}?\n"
                       f"{opts}\nAnswer with the letter of one option only.")
    lps = sc.score(prompts, [len(JUDGE_OPTS)] * len(prompts))
    for r, lp in zip(rows, lps):
        k = max(range(len(lp)), key=lambda i: lp[i])
        r["judge"] = JUDGE_CODE[k]
        r["judge_p"] = f"{math.exp(lp[k]):.3f}"


# ------------------------------------------------------------- audit score

AUDIT_CODES = {"t": "treat", "c": "contra", "w": "warn", "m": "none", "n": "none"}
RECALL_CODES = {"y": "Y", "b": "B", "n": "N"}


def label_columns(rows):
    """Annotator columns: every '*_label' column (annotator1_label ...),
    or the single 'your_label' column of the blank sheet."""
    cols = [c for c in rows[0] if c.endswith("_label") and c != "your_label"]
    return cols or ["your_label"]


def majority(votes):
    """Majority label of one item; None on a tie for first place."""
    c = Counter(v for v in votes if v).most_common()
    if not c:
        return None
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None
    return c[0][0]


def fleiss(items):
    """Fleiss' kappa over items that every annotator labeled."""
    items = [v for v in items if v and all(v)]
    if not items:
        return float("nan")
    n = len(items[0])
    tot, P = Counter(), []
    for v in items:
        c = Counter(v)
        tot.update(c)
        P.append((sum(x * x for x in c.values()) - n) / (n * (n - 1)))
    pbar = sum(P) / len(P)
    pe = sum((t / (len(items) * n)) ** 2 for t in tot.values())
    return (pbar - pe) / (1 - pe) if pe < 1 else 1.0


def agreement_block(votes, cols, who="annotators"):
    from itertools import combinations
    print(f"  {len(cols)} {who}: Fleiss' kappa {fleiss(votes):.2f}")
    if len(cols) > 1:
        pk = [kappa([(v[x], v[y]) for v in votes])[0] for x, y in combinations(range(len(cols)), 2)]
        print(f"  pairwise Cohen's kappa {min(pk):.2f} to {max(pk):.2f}")
        un = sum(len(set(v)) == 1 for v in votes if all(v))
        print(f"  unanimous on {un} of {len(votes)} items")


def score_audit(a):
    rows = list(csv.DictReader(open(a.score_audit, encoding="utf-8")))
    key = {r["item"]: r for r in csv.DictReader(open(a.key, encoding="utf-8"))}
    cols = label_columns(rows)
    votes, done, ties = [], [], 0
    for r in rows:
        v = [AUDIT_CODES.get((r.get(c) or "").strip().lower()[:1]) for c in cols]
        votes.append(v)
        m = majority(v)
        if r["item"] not in key or not any(v):
            continue
        if m is None:
            ties += 1
            continue
        done.append((key[r["item"]], m, v))
    print(f"AUDIT: {len(rows)} sentences, {len(done)} with a majority label, {ties} ties left out\n")
    agreement_block(votes, cols)
    if not done:
        return
    print("\n  majority label by graph relation and corpus (% of named sentences):")
    print("  relation          corpus      n   treats  must-not  warning  no link   supports")
    sup_all = Counter()
    for rel in RELS:
        for corp in ("wikipedia", "label", "pubmed"):
            g = [m for k, m, _ in done if k["relation"] == rel and k["corpus"] == corp]
            if not g:
                continue
            n = len(g)
            want = ("treat",) if rel == "indication" else ("contra", "warn")
            sup = sum(m in want for m in g)
            sup_all[(rel, "y")] += sup
            sup_all[(rel, "n")] += n
            print(f"  {rel:<17} {corp:<10} {n:3d}  {100 * g.count('treat') / n:6.1f}%"
                  f"  {100 * g.count('contra') / n:7.1f}%  {100 * g.count('warn') / n:6.1f}%"
                  f"  {100 * g.count('none') / n:6.1f}%   {100 * sup / n:6.1f}%")
    for rel in RELS:
        n = sup_all[(rel, "n")]
        if n:
            print(f"  {rel:<17} all        {n:3d}  supports the graph relation: "
                  f"{100 * sup_all[(rel, 'y')] / n:.1f}%")
    print("\n  'supports': an indication sentence says the drug treats the disease; a")
    print("  contraindication sentence says it must not be used or warns about harm.")
    print("\n  automatic checks against the majority label:")
    for name in ("judge", "cue"):
        pairs = [(k.get(name, ""), m) for k, m, _ in done]
        k_, n_ = kappa(pairs)
        if n_:
            agree = sum(x == y for x, y in pairs if x) / n_
            # binary: does the check call the relation supported, like the majority?
            def sup(k, lab):
                want = ("treat",) if k["relation"] == "indication" else ("contra", "warn")
                return lab in want
            bp = [(sup(k, k.get(name, "")), sup(k, m)) for k, m, _ in done if k.get(name)]
            bag = sum(x == y for x, y in bp) / len(bp)
            bk = kappa([(str(x), str(y)) for x, y in bp])[0]
            print(f"  {name:<6} four-way agreement {100 * agree:5.1f}% (kappa {k_:.2f});"
                  f"  supported/not agreement {100 * bag:5.1f}% (kappa {bk:.2f}), n={n_}")


def score_recall(a):
    rows = list(csv.DictReader(open(a.score_recall, encoding="utf-8")))
    cols = label_columns(rows)
    votes = [[RECALL_CODES.get((r.get(c) or "").strip().lower()[:1]) for c in cols] for r in rows]
    maj = [majority(v) for v in votes]
    n = sum(m is not None for m in maj)
    c = Counter(m for m in maj if m)
    print(f"RECALL: {len(rows)} contraindications the matcher misses in the label's own section,")
    print(f"        {n} with a majority label, {len(rows) - n} ties left out\n")
    agreement_block(votes, cols)
    print()
    for k, txt in (("Y", "stated in other words (a matcher miss)"),
                   ("B", "only a broader group is stated"),
                   ("N", "not in the Contraindications section")):
        print(f"  {k}  {c[k]:3d}  {100 * c[k] / max(1, n):5.1f}%  {txt}")


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges", default="matched_edges.csv")
    ap.add_argument("--wiki-cache", default="wiki_cache")
    ap.add_argument("--pubmed-cache", default="pubmed_cache")
    ap.add_argument("--pubmed-terms", default=os.environ.get("PUBMED_TERMS", ""),
                    help="the balanced PubMed fragments, '|'-separated (config.sh)")
    ap.add_argument("--labels", default="label_cache")
    ap.add_argument("--names-cache", default="gbif_cache")
    ap.add_argument("--ancestors", default="mondo_child_parent.csv")
    ap.add_argument("--judge", default="", help="model for the sentence judge")
    ap.add_argument("--judge-n", type=int, default=0,
                    help="judge at most N rows per relation and corpus (0: all)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--audit", type=int, default=0)
    ap.add_argument("--recall", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="relcheck_sentences.csv")
    ap.add_argument("--reuse", action="store_true",
                    help="read --out instead of rebuilding it (adds the judge to it)")
    ap.add_argument("--score-audit", default="")
    ap.add_argument("--key", default="audit_key.csv")
    ap.add_argument("--score-recall", default="")
    a = ap.parse_args()

    terms = [t.strip() for t in a.pubmed_terms.split("|") if t.strip()]
    a.pubmed_suffix = ""
    if terms:
        import hashlib
        a.pubmed_suffix = "__q" + hashlib.sha1("|".join(terms).encode("utf-8")).hexdigest()[:8]

    if a.score_audit or a.score_recall:
        if a.score_audit:
            score_audit(a)
        if a.score_recall:
            print()
            score_recall(a)
        return

    edges = [r for r in csv.DictReader(open(a.edges, encoding="utf-8"))
             if r["relation"] in RELS]
    anc = nt.load_ancestors(a.ancestors)

    def named_set(tgt):
        n = dict(nt.read_names(a.names_cache, "redir", tgt) or nt.EMPTY)
        n["synonyms"] = []
        n["ancestors"] = nt.climb(tgt, anc, 2)
        return nt.tier_sets(tgt, n)

    rows = []
    if a.reuse and os.path.exists(a.out):
        rows = list(csv.DictReader(open(a.out, encoding="utf-8")))
        print(f"read {len(rows):,} rows from {a.out}")
    else:
        cache = {}
        for e in edges:
            t = load_texts(a, e["source"], cache)
            sets = named_set(e["target"])
            for corp in CORPORA:
                text = t.get(corp)
                if text is None:
                    continue
                sents = matched_sentences(text, sets[2])
                joined = " ".join(sents)
                rows.append({
                    "id": e["id"], "relation": e["relation"], "drug": e["source"],
                    "disease": e["target"], "corpus": corp,
                    "named": int(bool(sents)), "n_sent": len(sents),
                    "cue_treat": int(bool(CUE_TREAT.search(joined))),
                    "cue_contra": int(bool(CUE_CONTRA.search(joined))),
                    "cue_warn": int(bool(CUE_WARN.search(joined))),
                    "sentence": (sents[0] if sents else "")[:600],
                    "judge": "", "judge_p": "",
                })
        print(f"built {len(rows):,} edge-corpus rows over {len(edges):,} edges")

    if a.judge:
        todo = [r for r in rows if int(r["named"]) and not r.get("judge")]
        if a.judge_n:
            rng = random.Random(a.seed)
            by = defaultdict(list)
            for r in todo:
                by[(r["relation"], r["corpus"])].append(r)
            todo = []
            for k in sorted(by):
                v = by[k]
                rng.shuffle(v)
                todo += v[:a.judge_n]
        print(f"judging {len(todo):,} named sentences with {a.judge}")
        judge_rows(a, todo)

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")

    # ------------------------------------------------------------ report
    def supported(r, how):
        if not int(r["named"]):
            return False
        if how == "named":
            return True
        if how == "cue":
            return bool(int(r["cue_treat"])) if r["relation"] == "indication" \
                else bool(int(r["cue_contra"]) or int(r["cue_warn"]))
        if how == "cue_strict":
            return bool(int(r["cue_treat"])) if r["relation"] == "indication" \
                else bool(int(r["cue_contra"]))
        if how == "judge":
            want = ("treat",) if r["relation"] == "indication" else ("contra", "warn")
            return r.get("judge") in want
        if how == "judge_strict":
            want = "treat" if r["relation"] == "indication" else "contra"
            return r.get("judge") == want
        return False

    # sanity check: does the sentence finder agree with the paper's tiers?
    ref = {"wikipedia": (a.edges, "tier"), "pubmed": ("me_bal_union.csv", "tier_second"),
           "label": ("me_label_any.csv", "tier")}
    print("\n  agreement with the paper's named/not-named call (same edges):")
    for corp, (f, col) in ref.items():
        if not os.path.exists(f):
            print(f"    {corp:<10} {f} not found, skipped")
            continue
        tiers = {r["id"]: int(r[col]) <= 2 for r in csv.DictReader(open(f, encoding="utf-8"))
                 if r.get(col, "") != ""}
        pairs = [(bool(int(r["named"])), tiers[r["id"]]) for r in rows
                 if r["corpus"] == corp and r["id"] in tiers]
        if pairs:
            ag = sum(x == y for x, y in pairs) / len(pairs)
            print(f"    {corp:<10} {100 * ag:5.1f}% of {len(pairs):,} edges")

    hows = ["named", "cue", "cue_strict"]
    judged_all = bool(rows) and all(r.get("judge") for r in rows if int(r["named"]))
    if judged_all:
        hows += ["judge", "judge_strict"]
    print("\n" + "=" * 78)
    print("DOES THE NAMING SENTENCE STATE THE RELATION?")
    print("=" * 78)
    print("  named         the text names the disease (the paper's measure)")
    print("  cue           ... and the sentence has a relation word; for a")
    print("                contraindication a warning word also counts")
    print("  cue_strict    ... contraindication needs contraindicated/must not/avoid ...")
    if judged_all:
        print("  judge         ... and the model judge reads the relation (warning counts)")
        print("  judge_strict  ... the judge reads 'must not be used'")
    print(f"\n  {'corpus':<10} {'definition':<13} {'ind %':>7} {'contra %':>9}"
          f"   OR [95% RBG CI]          p")
    for corp in CORPORA:
        rs = [r for r in rows if r["corpus"] == corp]
        if not rs:
            continue
        for how in hows:
            cells = defaultdict(lambda: {"indication": [0, 0], "contraindication": [0, 0]})
            k = Counter()
            for r in rs:
                y = supported(r, how)
                cells[r["drug"]][r["relation"]][0 if y else 1] += 1
                k[(r["relation"], y)] += 1
            rate = {rel: 100 * k[(rel, True)] / max(1, k[(rel, True)] + k[(rel, False)])
                    for rel in RELS}
            orm, ci, p = mh_rbg(cells)
            print(f"  {corp:<10} {how:<13} {rate['indication']:7.2f} "
                  f"{rate['contraindication']:9.2f}   {orm:5.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
                  f"   {p:.1e}")
        print()
    print("  Each row is the same edges; only the definition of 'stated' changes.")
    print("  The prose gap survives a stricter definition if OR stays above 1.")

    # support rate among named, per relation and corpus
    print("\n  share of NAMED facts whose sentence states the relation (cue / judge):")
    for corp in CORPORA:
        for rel in RELS:
            nm = [r for r in rows if r["corpus"] == corp and r["relation"] == rel
                  and int(r["named"])]
            if not nm:
                continue
            c = sum(supported(r, "cue") for r in nm) / len(nm)
            jd = [r for r in nm if r.get("judge")]
            js = (f"{100 * sum(supported(r, 'judge') for r in jd) / len(jd):5.1f}% (n={len(jd)})"
                  if jd else "   -")
            print(f"    {corp:<10} {rel:<17} named {len(nm):5d}   cue {100 * c:5.1f}%   judge {js}")

    # ---------------------------------------------------------- sheets
    rng = random.Random(a.seed)
    if a.audit:
        pick = []
        for rel in RELS:
            for corp in ("wikipedia", "label"):
                g = [r for r in rows if r["relation"] == rel and r["corpus"] == corp
                     and int(r["named"])]
                rng.shuffle(g)
                pick += g[:a.audit]
        rng.shuffle(pick)
        with open("audit_sheet.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["item", "drug", "disease", "sentence", "your_label", "note"])
            for i, r in enumerate(pick, 1):
                w.writerow([i, r["drug"], r["disease"], r["sentence"], "", ""])
        with open("audit_key.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["item", "id", "relation", "corpus", "cue", "judge"])
            for i, r in enumerate(pick, 1):
                cue = ("treat" if int(r["cue_treat"]) else
                       "contra" if int(r["cue_contra"]) else
                       "warn" if int(r["cue_warn"]) else "none")
                w.writerow([i, r["id"], r["relation"], r["corpus"], cue, r.get("judge", "")])
        print(f"\nwrote audit_sheet.csv ({len(pick)} sentences, relation hidden) and audit_key.csv")
        print("  your_label: T treats, C must not be used, W may cause/worsen/caution, M mention only")

    if a.recall:
        cache = {}
        cand = []
        for e in edges:
            if e["relation"] != "contraindication":
                continue
            t = load_texts(a, e["source"], cache)
            sec = t.get("label_contra") or ""
            if len(sec.strip()) < 30:
                continue
            sets = named_set(e["target"])
            if any(nt.mentions(sec, p) for p in sets[3]):
                continue
            cand.append((e, sec, sets[3][1:6]))
        rng.shuffle(cand)
        with open("recall_sheet.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["item", "drug", "disease", "family_names", "contraindications_section",
                        "your_label", "wording_in_label"])
            for i, (e, sec, fam) in enumerate(cand[:a.recall], 1):
                w.writerow([i, e["source"], e["target"], "; ".join(fam),
                            re.sub(r"\s+", " ", sec)[:1500], "", ""])
        print(f"wrote recall_sheet.csv ({min(a.recall, len(cand))} of {len(cand):,} "
              f"unmatched contraindications with a non-empty section)")
        print("  your_label: Y stated in other words, B only a broader group, N not there")


if __name__ == "__main__":
    main()
