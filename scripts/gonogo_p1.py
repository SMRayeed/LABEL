#!/usr/bin/env python3
"""
GO/NO-GO for proposal 1: does grounding tier predict LLM recall after the
frequency predictors are controlled for?

The claim under test is NOT "tier correlates with accuracy" -- it will, because
tier correlates with how much was written about the pair. The claim is that
tier carries signal the counts do not. So the deciding number is the partial
effect of tier in a model that already contains co-occurrence, page views,
object-set size and the answer-type prior. A univariate correlation here would
be the same mistake a reviewer would catch.

Three stages, because the inference runs separately on the cluster:

  build    read the tier caches, assemble a per-EDGE table with covariates,
           and write the probe prompts
  score    merge the model's responses back and mark each probe right or wrong
  regress  fit, and print GO or NO-GO

    python3 gonogo_p1.py build \\
        --edges pkg_ind_d2d.csv pkg_con_d2dis.csv pkg_off_d2dis.csv \\
        --names redirects --any-names --ancestors mondo_child_parent.csv \\
        --ancestor-depth 1 --n 600 \\
        --out-edges p1_edges.csv --out-prompts p1_prompts.jsonl

    # run p1_prompts.jsonl through a model; write {"id":..., "response":...}
    # per line to p1_responses.jsonl

    python3 gonogo_p1.py score --edges p1_edges.csv \\
        --prompts p1_prompts.jsonl --responses p1_responses.jsonl \\
        --out p1_scored.csv

    python3 gonogo_p1.py regress --scored p1_scored.csv

Needs name_tiers.py in the same directory, and numpy. Nothing else.
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import name_tiers as nt
except ImportError:
    sys.exit("name_tiers.py must sit in the same directory as this script")

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

PAGEVIEWS = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
             "/en.wikipedia/all-access/user/{title}/monthly/{start}/{end}")

# One question per relation. The target entity is never named in the open
# form; the verification form names it and asks for a judgment. Both are
# needed: open generation alone cannot distinguish "the model does not hold
# the fact" from "the model did not produce it under this phrasing", which is
# the confound Calderon et al. document.
TEMPLATES = {
    "indication": {
        "open": "Name one disease that {s} is used to treat. Answer with the disease name only.",
        "verify": "Is {s} used to treat {t}? Answer yes or no.",
    },
    "contraindication": {
        "open": "Name one condition in which {s} is contraindicated. Answer with the condition name only.",
        "verify": "Is {s} contraindicated in {t}? Answer yes or no.",
    },
    "off-label use": {
        "open": "Name one condition for which {s} is used off-label. Answer with the condition name only.",
        "verify": "Is {s} used off-label for {t}? Answer yes or no.",
    },
    "eats": {
        "open": "Name one species that {s} eats. Answer with the scientific name only.",
        "verify": "Does {s} eat {t}? Answer yes or no.",
    },
    "preysOn": {
        "open": "Name one species that {s} preys on. Answer with the scientific name only.",
        "verify": "Does {s} prey on {t}? Answer yes or no.",
    },
    "pollinates": {
        "open": "Name one plant species that {s} pollinates. Answer with the scientific name only.",
        "verify": "Does {s} pollinate {t}? Answer yes or no.",
    },
    "visitsFlowersOf": {
        "open": "Name one plant whose flowers {s} visits. Answer with the scientific name only.",
        "verify": "Does {s} visit the flowers of {t}? Answer yes or no.",
    },
}
GENERIC_TEMPLATE = {
    "open": "Complete the fact: {s} — {r} — ? Answer with the entity name only.",
    "verify": "Is this true: {s} {r} {t}? Answer yes or no.",
}


def template_for(rel):
    return TEMPLATES.get(rel, GENERIC_TEMPLATE)


# ----------------------------------------------------------------- stage 1

def pageviews(title, cache_dir, fx, months=12):
    """Mean monthly page views, cached. 0.0 when the article has none."""
    hit = nt.read_names(cache_dir, "pv", title)
    if hit is not None:
        return hit.get("mean", 0.0)
    now = time.gmtime()
    end = f"{now.tm_year}{now.tm_mon:02d}01"
    y, m = now.tm_year, now.tm_mon - months
    while m <= 0:
        m += 12
        y -= 1
    start = f"{y}{m:02d}01"
    url = PAGEVIEWS.format(title=urllib.parse.quote(title.replace(" ", "_"), safe=""),
                           start=start, end=end)
    body = fx.get(url)
    vals = [it.get("views", 0) for it in ((body or {}).get("items") or [])]
    mean = sum(vals) / len(vals) if vals else 0.0
    if body is not None:                      # only cache a real answer
        nt.write_names(cache_dir, "pv", title, {"mean": mean, "n": len(vals)})
    return mean


def build(a):
    rels = [r.strip() for r in a.relations.split(",") if r.strip()]
    mode = "none" if a.no_gbif else a.names
    anc = nt.load_ancestors(a.ancestors, a.ancestor_cols, a.flip_ancestors)
    fx = nt.Fetcher(rps=a.rps, contact=a.contact)
    rng = random.Random(a.seed)

    by_rel = nt.load_edges(a.edges, a.col_source, a.col_target, a.col_relation,
                           rels or None, binomials=a.binomials)
    if not by_rel:
        sys.exit("no edges read; check --col-* flags with name_tiers.py --inspect")
    if not rels:
        rels = [r for r, _ in sorted(by_rel.items(), key=lambda kv: -len(kv[1]))][:12]

    def names_for(e):
        if mode == "gbif":
            n = nt.read_names(a.gbif_cache, "gbif", e)
            if not n:
                return nt.EMPTY
            out = []
            for rk in (n.get("ancestors") or [])[:a.ancestor_depth]:
                out.append({"rank": rk.get("rank", ""),
                            "scientific": rk.get("scientific", ""),
                            "vernacular": (rk.get("vernacular") or [])[:4]})
            return {"synonyms": n.get("synonyms", []),
                    "vernacular": n.get("vernacular", []), "ancestors": out}
        if mode == "redirects":
            n = dict(nt.read_names(a.gbif_cache, "redir", e) or nt.EMPTY)
            n["synonyms"] = []
            n["ancestors"] = nt.climb(e, anc, a.ancestor_depth)
            return n
        return nt.EMPTY

    # object-set statistics, computed over the WHOLE relation rather than the
    # measured sample, since they are properties of the relation
    out_degree, obj_counts = {}, {}
    for rel in rels:
        edges = by_rel.get(rel, [])
        d = defaultdict(set)
        c = Counter()
        for s, t in edges:
            d[s].add(t)
            c[t] += 1
        out_degree[rel] = {k: len(v) for k, v in d.items()}
        obj_counts[rel] = c

    # Pass 1: choose the measurable edges per relation and fetch what they
    # need. Nothing per-edge is computed yet, because the co-occurrence index
    # has to see the whole sample before any single edge can be scored.
    chosen, stats = {}, {}
    for rel in rels:
        edges = list(by_rel.get(rel, []))
        if not edges:
            continue
        rng.shuffle(edges)
        cand = edges[:max(a.n, int(a.n * a.oversample))]
        print(f"  {rel}: fetching up to {len(cand):,} source articles",
              file=sys.stderr)
        nt.fetch_articles(fx, [s for s, _ in cand], a.wiki_cache)
        measurable = []
        for s_, t_ in cand:
            txt = nt.read_article(a.wiki_cache, s_)
            if txt is not None and len(txt) >= a.min_chars:
                measurable.append((s_, t_))
            if len(measurable) >= a.n:
                break
        if len(measurable) < 20:
            print(f"  {rel}: only {len(measurable)} measurable, skipping",
                  file=sys.stderr)
            continue
        tgts = sorted({t for _, t in measurable})
        if mode == "redirects":
            nt.fetch_redirect_names(fx, tgts, a.gbif_cache)
        elif mode == "gbif":
            nt.fetch_gbif_names(fx, tgts, a.gbif_cache,
                                ancestor_vernaculars=a.anc_vern)
        counts = obj_counts[rel]
        total = sum(counts.values()) or 1
        ent = -sum((v / total) * math.log(v / total) for v in counts.values())
        modal = counts.most_common(1)[0][1] / total if counts else 0.0
        partners = defaultdict(set)
        for s_, t_ in by_rel[rel]:
            partners[s_].add(t_)
        chosen[rel] = measurable
        stats[rel] = {"ent": ent, "modal": modal, "partners": partners,
                      "targets": tgts}
        print(f"  {rel}: {len(measurable):,} measurable edges", file=sys.stderr)

    if not chosen:
        sys.exit("no relation had enough measurable edges")

    # Pass 2: one co-occurrence index over every source article in the sample.
    all_src = sorted({s_ for m in chosen.values() for s_, _ in m})
    all_ent = sorted({x for m in chosen.values() for e in m for x in e})
    print(f"\nbuilding the co-occurrence index: {len(all_ent):,} entities "
          f"over {len(all_src):,} articles", file=sys.stderr)
    t0 = time.time()
    idx = cooc_index(all_ent, all_src, a.wiki_cache)
    print(f"  index built in {time.time() - t0:.0f}s", file=sys.stderr)

    if a.pageviews:
        need = all_src
        print(f"fetching page views for {len(need):,} sources "
              f"(one request each; ^C is safe, it resumes from cache)",
              file=sys.stderr)
        t0 = time.time()
        for i, s_ in enumerate(need):
            pageviews(s_, a.gbif_cache, fx)
            if (i + 1) % 25 == 0:
                rate = (i + 1) / max(1e-9, time.time() - t0)
                left = (len(need) - i - 1) / max(1e-9, rate)
                print(f"    {i + 1}/{len(need)}  ~{left / 60:.0f} min left",
                      end="\r", file=sys.stderr)
        print(" " * 46, end="\r", file=sys.stderr)

    # Pass 3: the per-edge rows, all from memory.
    rows, prompts = [], []
    for rel, measurable in chosen.items():
        st = stats[rel]
        partners = st["partners"]
        for i, (src, tgt) in enumerate(measurable):
            text = nt.read_article(a.wiki_cache, src)
            sets = nt.tier_sets(tgt, names_for(tgt))
            hits = nt.hit_tiers(text, sets)
            tier = next((k for k, h in enumerate(hits) if h), 4)
            cooc = len(idx.get(src, set()) & idx.get(tgt, set()))
            pv = pageviews(src, a.gbif_cache, fx) if a.pageviews else 0.0
            eid = f"{rel}|{i}"
            rows.append({
                "id": eid, "relation": rel, "source": src, "target": tgt,
                "tier": tier,
                "category_only": int(tier >= 3),
                "untraced": int(tier == 4),
                "art_len": len(text),
                "log_art_len": f"{math.log(len(text) + 1):.4f}",
                "pageviews": f"{pv:.1f}",
                "log_pageviews": f"{math.log(pv + 1):.4f}",
                "cooccurrence": cooc,
                "log_cooccurrence": f"{math.log(cooc + 1):.4f}",
                "out_degree": len(partners[src]),
                "log_out_degree": f"{math.log(len(partners[src]) + 1):.4f}",
                "obj_entropy": f"{st['ent']:.4f}",
                "answer_prior": f"{st['modal']:.6f}",
                "n_valid_answers": len(partners[src]),
            })
            tpl = template_for(rel)
            prompts.append({"id": eid, "kind": "open", "relation": rel,
                            "prompt": tpl["open"].format(s=src, r=rel, t=tgt)})
            prompts.append({"id": eid, "kind": "verify_pos", "relation": rel,
                            "prompt": tpl["verify"].format(s=src, r=rel, t=tgt)})
            for _try in range(12):
                neg = rng.choice(st["targets"])
                if neg not in partners[src] and neg.lower() != src.lower():
                    break
            else:
                neg = None
            if neg:
                prompts.append({"id": eid, "kind": "verify_neg", "relation": rel,
                                "negative": neg,
                                "prompt": tpl["verify"].format(s=src, r=rel, t=neg)})

    if not rows:
        sys.exit("nothing built")
    with open(a.out_edges, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(a.out_prompts, "w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps(p) + "\n")

    print(f"\nwrote {a.out_edges}: {len(rows):,} edges")
    print(f"wrote {a.out_prompts}: {len(prompts):,} prompts")
    by_tier = Counter(r["tier"] for r in rows)
    print("\ntier distribution (0 exact, 1 synonym, 2 common name, "
          "3 ancestor, 4 none):")
    for k in range(5):
        n = by_tier.get(k, 0)
        print(f"  tier {k}: {n:>6,}  {100 * n / len(rows):>5.1f}%")
    if by_tier.get(4, 0) + by_tier.get(3, 0) < 30:
        print("\nWARNING: fewer than 30 edges at tier 3 or 4. The regression "
              "has almost no\n         variation to work with. Raise --n or "
              "add relations.")
    print(f"\nnetwork: {fx.report()}")


def normalize_grams(text, max_len=3):
    """Lowercase 1..max_len word n-grams of a document, as a set.

    Built once per article and thrown away, so memory stays bounded by the
    single largest article rather than by the corpus."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    grams = set()
    n = len(words)
    for i in range(n):
        g = words[i]
        grams.add(g)
        for k in range(1, max_len):
            if i + k < n:
                g = g + " " + words[i + k]
                grams.add(g)
    return grams


def cooc_index(entities, articles, wiki_cache):
    """entity -> set of article ids naming it, in ONE pass over the corpus.

    The earlier version rescanned the whole cache per edge, which is
    quadratic and does not finish. Here each article is read once, reduced to
    its n-gram set, and tested against every entity; the corpus is the source
    articles of the measured edges, which is the corpus the tier measurement
    already used."""
    norm = {}
    for e in entities:
        key = " ".join(re.findall(r"[a-z0-9]+", e.lower()))
        if key:
            norm.setdefault(key, []).append(e)
    keys = list(norm)
    long_keys = [k for k in keys if len(k.split()) > 3]
    short_keys = [k for k in keys if len(k.split()) <= 3]
    idx = defaultdict(set)
    for i, name in enumerate(articles):
        txt = nt.read_article(wiki_cache, name)
        if not txt:
            continue
        grams = normalize_grams(txt)
        for k in short_keys:
            if k in grams:
                for e in norm[k]:
                    idx[e].add(i)
        if long_keys:
            low = txt.lower()
            for k in long_keys:
                if k in low:
                    for e in norm[k]:
                        idx[e].add(i)
        if (i + 1) % 100 == 0:
            print(f"    co-occurrence index {i + 1}/{len(articles)}",
                  end="\r", file=sys.stderr)
    print(" " * 46, end="\r", file=sys.stderr)
    return idx


# ----------------------------------------------------------------- stage 2

def score(a):
    for path, what in ((a.edges, "the build stage"), (a.prompts, "the build stage")):
        if not os.path.exists(path):
            sys.exit(f"{path} is missing. Run {what} first.")
    if not os.path.exists(a.responses):
        sys.exit(
            f"{a.responses} is missing.\n\n"
            f"That file is the model's output and this script does not produce "
            f"it. Run the\nprompts in {a.prompts} through a model and write one "
            f"JSON object per line:\n\n"
            f'  {{"id": "<the prompt id>", "kind": "<open|verify_pos|verify_neg>", '
            f'"response": "<text>"}}\n\n'
            f"run_prompts.py does this with a local transformers model if you "
            f"want a ready one.")
    edges = {r["id"]: r for r in csv.DictReader(open(a.edges, encoding="utf-8"))}
    prompts = defaultdict(dict)
    with open(a.prompts, encoding="utf-8") as fh:
        for line in fh:
            p = json.loads(line)
            prompts[p["id"]][p["kind"]] = p
    resp = {}
    with open(a.responses, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rid, kind = r.get("id"), r.get("kind")
            if kind is None:                   # allow "id|kind" in one field
                if rid and "#" in rid:
                    rid, kind = rid.rsplit("#", 1)
            resp[(rid, kind)] = (r.get("response") or "").strip()

    out, missing = [], 0
    for eid, row in edges.items():
        tgt = row["target"]
        o = resp.get((eid, "open"))
        vp = resp.get((eid, "verify_pos"))
        vn = resp.get((eid, "verify_neg"))
        if o is None and vp is None:
            missing += 1
            continue
        row = dict(row)
        row["open_correct"] = ""
        if o is not None:
            row["open_correct"] = int(nt.mentions(o, tgt, "word"))
        row["verify_pos_yes"] = "" if vp is None else int(_yes(vp))
        row["verify_neg_yes"] = "" if vn is None else int(_yes(vn))
        if vp is not None and vn is not None:
            # a model that says yes to everything is not demonstrating
            # knowledge; discrimination needs both items
            row["verify_correct"] = int(_yes(vp) and not _yes(vn))
        else:
            row["verify_correct"] = ""
        out.append(row)

    if not out:
        sys.exit("no responses matched the prompt ids")
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {a.out}: {len(out):,} edges scored, {missing:,} without responses")
    for col in ("open_correct", "verify_correct"):
        vals = [int(r[col]) for r in out if r[col] != ""]
        if vals:
            print(f"  {col:<16} {100 * sum(vals) / len(vals):>6.2f}%  (n={len(vals):,})")


def _yes(text):
    t = text.strip().lower()
    return t.startswith("yes") or t.startswith("y ") or t == "y" or "yes" in t[:12]


# ----------------------------------------------------------------- stage 3

def logistic(X, y, names, tol=1e-8, iters=100):
    """IRLS logistic regression with Wald standard errors. numpy only."""
    import numpy as np
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / W
        XtW = X.T * W
        try:
            new = np.linalg.solve(XtW @ X + 1e-10 * np.eye(k), XtW @ z)
        except np.linalg.LinAlgError:
            return None
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    W = np.clip(mu * (1 - mu), 1e-9, None)
    try:
        cov = np.linalg.inv((X.T * W) @ X + 1e-10 * np.eye(k))
    except np.linalg.LinAlgError:
        return None
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    zstat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p = [math.erfc(abs(float(t)) / math.sqrt(2)) for t in zstat]
    ll = float(np.sum(y * np.log(np.clip(mu, 1e-12, 1)) +
                      (1 - y) * np.log(np.clip(1 - mu, 1e-12, 1))))
    return {"names": names, "beta": beta, "se": se, "z": zstat, "p": p,
            "ll": ll, "n": n}


def regress(a):
    try:
        import numpy as np
    except ImportError:
        sys.exit("this stage needs numpy: pip install numpy")

    if not os.path.exists(a.scored):
        sys.exit(f"{a.scored} is missing. Run the score stage first, which "
                 f"needs the model's\nresponses -- see the score stage's own "
                 f"message for the format.")
    rows = [r for r in csv.DictReader(open(a.scored, encoding="utf-8"))]
    dv = a.outcome
    rows = [r for r in rows if r.get(dv) not in (None, "")]
    if len(rows) < 50:
        sys.exit(f"only {len(rows)} rows have {dv}; need at least 50")

    y = [float(r[dv]) for r in rows]
    rels = sorted({r["relation"] for r in rows})
    tier_n = Counter(int(r["tier"]) for r in rows)

    # The claim is that a fact stated only at category level is harder to
    # recall than one stated specifically. "Not stated at all" is a different
    # condition, and when it dominates the sample the pooled tier>=3 split
    # stops testing the claim. So the primary test runs on edges whose text
    # says SOMETHING about the target -- tiers 0 to 3 -- and asks whether
    # tier 3 alone predicts failure.
    with_text = [i for i, r in enumerate(rows) if int(r["tier"]) <= 3]
    share_t4 = 100 * tier_n.get(4, 0) / len(rows)
    covs = ["log_cooccurrence", "log_pageviews", "log_out_degree",
            "obj_entropy", "answer_prior", "log_art_len"]
    covs = [c for c in covs if c in rows[0] and
            len({r[c] for r in rows}) > 1]

    dropped_collinear = []

    def design(include_tier, include_covs, include_rel, idx=None, var="category_only"):
        """Returns (X, y, names). `idx` restricts to a subset of rows."""
        sel = list(range(len(rows))) if idx is None else idx
        rr = [rows[i] for i in sel]
        yy = [y[i] for i in sel]
        names, cols = ["intercept"], [[1.0] * len(rr)]
        if include_tier:
            names.append(var)
            if var == "category_only":
                cols.append([float(r["category_only"]) for r in rr])
            else:                                  # tier 3 exactly
                cols.append([1.0 if int(r["tier"]) == 3 else 0.0 for r in rr])
        present = sorted({r["relation"] for r in rr})
        if include_covs:
            for c in covs:
                v = [float(r[c]) for r in rr]
                if len(set(v)) < 2:
                    continue
                # A covariate computed per relation takes one value per
                # relation, so with relation dummies in the model it is a
                # linear combination of them and the design goes rank
                # deficient. The ridge term then hides it behind standard
                # errors in the tens of thousands instead of failing.
                if include_rel and len(present) > 1:
                    per_rel = {r["relation"]: set() for r in rr}
                    for r, x in zip(rr, v):
                        per_rel[r["relation"]].add(round(x, 9))
                    if all(len(u) == 1 for u in per_rel.values()):
                        dropped_collinear.append(c)
                        continue
                m, sd = sum(v) / len(v), (np.std(v) or 1.0)
                names.append(c)
                cols.append([(x - m) / sd for x in v])
        if include_rel:
            if len(present) > 1:
                for rl in present[1:]:
                    names.append(f"rel[{rl}]")
                    cols.append([1.0 if r["relation"] == rl else 0.0 for r in rr])
        return np.array(cols).T, yy, names

    base_rate = sum(y) / len(y)

    print("=" * 78)
    print(f"GO/NO-GO — proposal 1    outcome: {dv}    n = {len(rows):,}")
    print("=" * 78)
    print(f"base rate {100 * base_rate:.2f}%   relations: {', '.join(rels)}")
    print(f"covariates offered: {', '.join(covs)}")

    _ = design(True, True, True)          # populate the drop list
    if dropped_collinear:
        uniq = sorted(set(dropped_collinear))
        print(f"dropped as collinear with the relation dummies: "
              f"{', '.join(uniq)}")
        print("  (each is constant within a relation, so the dummies already "
              "carry it)")
    print()
    print("accuracy by tier")
    by = defaultdict(list)
    for r, yy in zip(rows, y):
        by[int(r["tier"])].append(yy)
    labels = {0: "exact name", 1: "+synonym", 2: "+common name",
              3: "ancestor only", 4: "not named at all"}
    for k in range(5):
        v = by.get(k, [])
        if v:
            mid, lo, hi = nt.wilson(int(sum(v)), len(v))
            print(f"  tier {k} {labels[k]:<18} {mid:>6.2f}% [{lo:>5.2f}-{hi:>5.2f}]"
                  f"  n={len(v):>5,}")
    cat = [yy for r, yy in zip(rows, y) if int(r["category_only"])]
    non = [yy for r, yy in zip(rows, y) if not int(r["category_only"])]
    if not cat or not non:
        sys.exit("\nNO-GO (degenerate): every edge falls on one side of the "
                 "category_only split,\nso there is no contrast to fit.")
    print(f"\n  named specifically  {100 * sum(non) / len(non):>6.2f}%  n={len(non):,}")
    print(f"  category or absent  {100 * sum(cat) / len(cat):>6.2f}%  n={len(cat):,}")
    print(f"  raw difference      {100 * (sum(non) / len(non) - sum(cat) / len(cat)):>6.2f} points")

    uni = logistic(*design(True, False, False))
    full = logistic(*design(True, True, True))
    nod = logistic(*design(False, True, True))
    if full is None or uni is None:
        sys.exit("\nthe model did not converge; check for constant columns")

    print("\n" + "-" * 78)
    print("SECONDARY: all edges, category_only = tier 3 or 4 "
          f"({share_t4:.0f}% of those are tier 4)")
    print("-" * 78)
    print(f"{'term':<22} {'coef':>9} {'se':>8} {'z':>7} {'p':>10}")
    for i, nm in enumerate(full["names"]):
        print(f"{nm:<22} {full['beta'][i]:>9.4f} {full['se'][i]:>8.4f} "
              f"{full['z'][i]:>7.2f} {full['p'][i]:>10.2e}")

    ti = full["names"].index("category_only")
    coef, pval = float(full["beta"][ti]), float(full["p"][ti])
    uni_coef = float(uni["beta"][uni["names"].index("category_only")])
    shrink = (1 - coef / uni_coef) * 100 if uni_coef else float("nan")

    # ----------------------------------------------------------- primary
    n_t3 = tier_n.get(3, 0)
    n_spec = sum(tier_n.get(k, 0) for k in (0, 1, 2))
    print("\n" + "-" * 78)
    print("PRIMARY: edges whose text names the target at SOME level "
          "(tiers 0-3)")
    print("-" * 78)
    print(f"category stated only (tier 3): {n_t3:,}    "
          f"stated specifically (tiers 0-2): {n_spec:,}")

    r_coef = r_p = r_shrink = float("nan")
    r_full = None
    if n_t3 < 25 or n_spec < 25:
        print("\nToo few edges on one side for the restricted test. The "
              "secondary result above\nis all this sample supports, and it "
              "mostly compares 'named' against 'absent'.")
    else:
        sub_y = [y[i] for i in with_text]
        if len(set(sub_y)) < 2:
            print("\nThe outcome does not vary within this subset.")
        else:
            r_uni = logistic(*design(True, False, False, with_text, "tier3_only"))
            r_full = logistic(*design(True, True, True, with_text, "tier3_only"))
            if r_full is None:
                print("\nThe restricted model did not converge.")
            else:
                print(f"\n{'term':<22} {'coef':>9} {'se':>8} {'z':>7} {'p':>10}")
                for i, nm in enumerate(r_full["names"]):
                    print(f"{nm:<22} {r_full['beta'][i]:>9.4f} "
                          f"{r_full['se'][i]:>8.4f} {r_full['z'][i]:>7.2f} "
                          f"{r_full['p'][i]:>10.2e}")
                rti = r_full["names"].index("tier3_only")
                r_coef = float(r_full["beta"][rti])
                r_p = float(r_full["p"][rti])
                r_uni_c = float(r_uni["beta"][r_uni["names"].index("tier3_only")])
                r_shrink = (1 - r_coef / r_uni_c) * 100 if r_uni_c else float("nan")
                print(f"\ntier3_only alone          coef {r_uni_c:>8.4f}")
                print(f"tier3_only with controls  coef {r_coef:>8.4f}   "
                      f"p = {r_p:.2e}")
                print(f"shrinkage from controls   {r_shrink:>8.1f}%")

    print("\n" + "-" * 78)
    print(f"secondary category_only alone         coef {uni_coef:>8.4f}")
    print(f"secondary category_only w/ controls   coef {coef:>8.4f}   "
          f"p = {pval:.2e}")
    print(f"secondary shrinkage                   {shrink:>8.1f}%")

    # the verdict runs on the restricted test when it exists
    if r_full is not None and not math.isnan(r_p):
        coef, pval, shrink, uni_coef = r_coef, r_p, r_shrink, r_uni_c
        primary = "tier3_only, among edges with text evidence"
    else:
        primary = "category_only, all edges"
    print(f"\nverdict is read off: {primary}")

    # An estimate whose interval spans a moderate effect in BOTH directions
    # carries no information, and a verdict read off it is noise dressed as a
    # decision. Check precision before judging sign or significance.
    se_t = float(full["se"][ti]) if r_full is None else float(r_full["se"][
        r_full["names"].index("tier3_only")])
    ci_lo, ci_hi = coef - 1.96 * se_t, coef + 1.96 * se_t
    n_small = min(len(cat), len(non))
    underpowered = (ci_lo < -0.5 and ci_hi > 0.5) or n_small < 30 or len(rows) < 200

    print("\n" + "=" * 78)
    if underpowered:
        print("UNDERPOWERED \u2014 no verdict")
        print("=" * 78)
        print(f"n = {len(rows):,}, smallest group = {n_small:,}, "
              f"coefficient {coef:+.3f} with standard error {se_t:.3f}.")
        print(f"The 95% interval runs [{ci_lo:+.2f}, {ci_hi:+.2f}], which is "
              f"compatible with a strong")
        print("effect in either direction at once. Nothing can be concluded "
              "from it.")
        print("\nThe tier table above is the part worth reading at this size. "
              "If the rates")
        print("fall as the text gets less specific, the hypothesis is alive "
              "and the sample")
        print("is simply too small to show it.")
        print("\nRun the full edge set rather than a --limit subset, and check "
              "that more than")
        print("one relation is present: a limited run takes the first rows, "
              "which may all")
        print("come from one relation.")
        print()
        print(f"network: nothing further to fetch; the probe is local.")
        return
    ok_sign = coef < 0
    survives = shrink <= 50.0
    if not ok_sign or pval >= 0.05:
        print("NO-GO")
        print("=" * 78)
        if not ok_sign:
            print("The effect runs the wrong way once the counts are controlled for.")
        else:
            print(f"The tier term is not significant with the counts in the model "
                  f"(p = {pval:.3f}).")
        print("Granularity was standing in for frequency. Put the month into")
        print("proposal 3, whose measurements are already complete.")
    elif shrink > 75.0:
        print("NO-GO")
        print("=" * 78)
        print(f"The coefficient falls {shrink:.0f}% when the counts enter, from "
              f"{uni_coef:.3f} to {coef:.3f}.")
        print(f"The residual p = {pval:.3f} is what is left of a frequency effect, "
              f"not evidence")
        print("of a separate one. A reviewer reads the shrinkage before the p-value.")
    elif pval < 0.01 and survives:
        print("GO")
        print("=" * 78)
        print("The tier term keeps a significant negative effect, and it survives")
        print("the count predictors rather than being absorbed by them. Being")
        print("stated only at category level predicts failure beyond frequency.")
        print("\nNext: the GloBI replication for a second domain, and the")
        print("model-size series for the scale prediction.")
    else:
        print("MARGINAL")
        print("=" * 78)
        if not survives:
            print(f"Right direction and significant, but the coefficient loses "
                  f"{shrink:.0f}% to the counts.")
        else:
            print(f"Right direction and it survives the controls, but p = {pval:.3f}.")
        print("Underpowered or partly confounded rather than absent. Add the GloBI")
        print("relations before deciding; three relations is thin for a")
        print("relation-level claim.")
    print()
    print("Decision rule: GO needs the right sign, p < 0.01, and under 50%")
    print("shrinkage. Over 75% shrinkage is NO-GO whatever the p-value.")
    print("Read the tier table above as well \u2014 a tier holding very few edges")
    print("makes the fit unstable however the numbers come out.")


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    b = sub.add_parser("build", help="edge table + covariates + prompts")
    b.add_argument("--edges", required=True, nargs="+")
    b.add_argument("--col-source", default="sourceTaxonName")
    b.add_argument("--col-target", default="targetTaxonName")
    b.add_argument("--col-relation", default="interactionTypeName")
    b.add_argument("--relations", default="")
    b.add_argument("--n", type=int, default=600)
    b.add_argument("--oversample", type=float, default=3.0)
    b.add_argument("--min-chars", type=int, default=3000)
    b.add_argument("--wiki-cache", default="wiki_cache")
    b.add_argument("--gbif-cache", default="gbif_cache")
    b.add_argument("--names", choices=["gbif", "redirects", "none"], default="gbif")
    b.add_argument("--no-gbif", action="store_true")
    b.add_argument("--ancestors", default="")
    b.add_argument("--ancestor-cols", default="")
    b.add_argument("--flip-ancestors", action="store_true")
    b.add_argument("--ancestor-depth", type=int, default=1)
    b.add_argument("--no-ancestor-vernaculars", dest="anc_vern",
                   action="store_false", default=True)
    b.add_argument("--binomials-only", dest="binomials", action="store_true",
                   default=True)
    b.add_argument("--any-names", dest="binomials", action="store_false")
    b.add_argument("--pageviews", action="store_true",
                   help="add the page-view covariate. One request per source "
                        "against a rate-limited host, so roughly a minute per "
                        "100 sources on the first run; cached after.")
    b.add_argument("--rps", type=float, default=3.0)
    b.add_argument("--contact", default="")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out-edges", default="p1_edges.csv")
    b.add_argument("--out-prompts", default="p1_prompts.jsonl")

    s = sub.add_parser("score", help="merge model responses")
    s.add_argument("--edges", required=True)
    s.add_argument("--prompts", required=True)
    s.add_argument("--responses", required=True)
    s.add_argument("--out", default="p1_scored.csv")

    r = sub.add_parser("regress", help="fit and print the verdict")
    r.add_argument("--scored", required=True)
    r.add_argument("--outcome", default="open_correct",
                   choices=["open_correct", "verify_correct", "mc_correct"],
                   help="mc_correct comes from run_probe_cpu.py, which scores "
                        "candidates by log-probability instead of generating")

    a = ap.parse_args()
    {"build": build, "score": score, "regress": regress}[a.stage](a)


if __name__ == "__main__":
    main()
