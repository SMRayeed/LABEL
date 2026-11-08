#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The neuro-symbolic step, measured on single edges.

An article that names only the parent category of an edge's target gives a
model almost nothing: the cliff test puts tier 3 beside silence rather than
beside naming. But it is not nothing to a system that can look the category
up. The category rules out every entity that is not one of its members, and
the ontology can list those members.

This measures whether that works. For each tier-3 edge it reads the matched
category off the article, asks the ontology for that category's members,
scores them, and takes the best one. The question is whether the true
target comes back.

Four conditions, every one of them choosing from a set of the same size k,
so the chance rate is 1/k throughout and the numbers are directly
comparable:

  ours          the ontology expansion of the category named on the page.
                The page is used symbolically, to find the category; it is
                not put in the prompt.
  ours_ctx      the same candidates, with the article in the prompt as
                well. Says whether reading the page adds anything once the
                category has been extracted from it.
  random_k      k candidates drawn from the relation's own target pool,
                with no article. The control for set size: a small set is
                easy whatever is in it, and this is what k alone buys.
  rag_pool      k drawn the same way, with the article in the prompt. This
                is naive retrieval at matched difficulty, and it is the
                baseline the method has to beat.

ours against rag_pool, paired, is the result. Both see k options and both
have the article available; the difference is that one of them spends the
page on finding a category and then asks an ontology, and the other hands
the page to the model.

Two accountings are reported, because the expansion can fail to contain the
target while a constructed control set always contains it:

  on recoverable edges   items where the expansion holds the target, so
                         every condition is choosing from a set that
                         contains it. The clean comparison.
  overall                every tier-3 edge, with an expansion that misses
                         the target counted against `ours`. The honest
                         end-to-end number.

    python3 recover.py --scored me_union.csv \\
        --wiki-cache wiki_cache --ancestors mondo_child_parent.csv \\
        --names-cache gbif_cache --dry-run

    python3 recover.py --scored me_union.csv \\
        --wiki-cache wiki_cache --ancestors mondo_child_parent.csv \\
        --names-cache gbif_cache \\
        --model Qwen/Qwen2.5-7B-Instruct --threads 12 --batch-size 8 \\
        --out recover_7B.csv

Needs name_tiers.py and router.py beside it, and torch plus transformers
unless --dry-run.
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import name_tiers as nt
except ImportError as e:
    sys.exit(f"name_tiers.py has to sit beside this script: {e}")
try:
    from router import Scorer, stem, expand, wilson, mcnemar
except ImportError as e:
    sys.exit(f"router.py has to sit beside this script: {e}")

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

CONDS = ["ours", "ours_ctx", "random_k", "rag_pool"]


def _drop_balanced(t, open_, close_):
    """Remove every outermost open_...close_ span, nesting included."""
    pat = re.compile(re.escape(open_) + "|" + re.escape(close_))
    out, depth, last, opened = [], 0, 0, 0
    for m in pat.finditer(t):
        if m.group() == open_:
            if depth == 0:
                out.append(t[last:m.start()])
                opened = m.start()
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0:
                last = m.end()
    out.append(t[opened:] if depth else t[last:])
    return "".join(out)


_PREFIXED = re.compile(r"\[\[\s*(?:file|image|category|datei)\s*:", re.I)


def _drop_prefixed_links(t):
    """Remove [[File:...]] and friends, whose captions can hold nested links."""
    out, i = [], 0
    while True:
        m = _PREFIXED.search(t, i)
        if not m:
            out.append(t[i:])
            break
        out.append(t[i:m.start()])
        depth, j = 0, m.start()
        while j < len(t):
            if t.startswith("[[", j):
                depth += 1
                j += 2
            elif t.startswith("]]", j):
                depth -= 1
                j += 2
                if depth == 0:
                    break
            else:
                j += 1
        i = j
    return "".join(out)


def clean_wikitext(t):
    """Wikipedia source to readable prose; plain text passes through.

    Templates (infoboxes, citation templates, maintenance tags) and tables
    are removed whole, references and comments are dropped, links keep
    their visible text, and headings keep their words. What remains is
    what a reader of the rendered article sees in the running text, which
    is what a retriever would index.
    """
    if not t or ("{{" not in t and "[[" not in t and "<ref" not in t):
        return t
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    t = re.sub(r"<ref[^>]*?/>", " ", t, flags=re.I)
    t = re.sub(r"<ref[^>]*?>.*?</ref\s*>", " ", t, flags=re.I | re.S)
    t = _drop_balanced(t, "{{", "}}")
    t = _drop_balanced(t, "{|", "|}")
    t = _drop_prefixed_links(t)
    t = re.sub(r"\[\[[^\[\]|]*\|([^\[\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[\[([^\[\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\]", " ", t)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.replace("&nbsp;", " ").replace("&ndash;", "-").replace("&mdash;", "-")
    t = re.sub(r"__[A-Z]+__", " ", t)
    t = re.sub(r"^\s*=+\s*(.*?)\s*=+\s*$", r"\1", t, flags=re.M)
    t = re.sub(r"^\s*[*#:;]+\s*", "", t, flags=re.M)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.strip()


def snippet(text, phrases, lead=800, win=280, cap=1800, clean=True):
    """The page as a retriever would hand it over.

    The opening of the article, plus a window around every mention of the
    given phrases, merged where they overlap and capped. A retriever
    returns the passage that matches the query, not the first six thousand
    characters, so this is both the fairer baseline and a quarter of the
    tokens, which is what makes the large model affordable on CPU.
    """
    if not text:
        return ""
    if clean:
        text = clean_wikitext(text)
    low = text.lower()
    spans = []
    for ph in phrases:
        ph = (ph or "").strip().lower()
        if len(ph) < 4:
            continue
        pat = r"\b" + r"\s+".join(re.escape(w) for w in ph.split()) + r"s?\b"
        for m in re.finditer(pat, low):
            spans.append((max(0, m.start() - win), min(len(text), m.end() + win)))
    spans.sort()
    merged = []
    for a_, b_ in spans:
        if merged and a_ <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b_))
        else:
            merged.append((a_, b_))
    parts = [text[:lead].strip()]
    for a_, b_ in merged:
        a_ = max(a_, lead)
        if b_ > a_:
            parts.append("... " + text[a_:b_].strip() + " ...")
    return "\n".join(parts)[:cap]


class PrefixScorer:
    """Scores many continuations of one prompt, encoding the prompt once.

    Every candidate for an edge shares the same prompt, and in the passage
    conditions that prompt is a few hundred tokens long. Scoring each
    (prompt, candidate) pair from scratch re-reads the passage once per
    candidate. This reads it once, keeps the attention cache, and scores the
    candidates against it, which on CPU is the difference between a day and
    a couple of hours at 7B.

    Cache objects differ between transformers versions, so on its first
    group it scores the same jobs both ways and compares. If they disagree,
    or anything raises, it says so and hands every job to the plain scorer
    for the rest of the run. Correctness never depends on the fast path.
    """

    def __init__(self, base, cont_batch=32):
        self.base = base
        self.cont_batch = cont_batch
        self.calls = 0
        self.ok = None             # None until the self-check has run

    @staticmethod
    def _kv(cache):
        if hasattr(cache, "layers"):
            return [(l.keys, l.values) for l in cache.layers]
        if hasattr(cache, "key_cache"):
            return list(zip(cache.key_cache, cache.value_cache))
        return [tuple(x[:2]) for x in cache]

    def _fresh_cache(self, kv, b):
        from transformers import DynamicCache
        c = DynamicCache()
        for i, (k, v) in enumerate(kv):
            c.update(k.expand(b, *k.shape[1:]).contiguous(),
                     v.expand(b, *v.shape[1:]).contiguous(), i)
        return c

    def _group(self, prompt, conts):
        torch, tok, model = self.base.torch, self.base.tok, self.base.model
        dev = self.base.device
        p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
        c_ids = []
        for c in conts:
            e = tok(c, add_special_tokens=False)["input_ids"]
            c_ids.append(e or [tok.eos_token_id])
        out = []
        with torch.no_grad():
            first = model(input_ids=torch.tensor([p_ids], device=dev),
                          use_cache=True)
            kv = self._kv(first.past_key_values)
            head = torch.log_softmax(first.logits[0, -1].float(), dim=-1)
            for b in range(0, len(c_ids), self.cont_batch):
                chunk = c_ids[b:b + self.cont_batch]
                n, w = len(chunk), max(len(x) for x in chunk)
                pad = tok.pad_token_id if tok.pad_token_id is not None else 0
                ids = torch.full((n, w), pad, dtype=torch.long)
                att = torch.zeros((n, len(p_ids) + w), dtype=torch.long)
                att[:, :len(p_ids)] = 1
                for i, x in enumerate(chunk):
                    ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
                    att[i, len(p_ids):len(p_ids) + len(x)] = 1
                pos = (len(p_ids) + torch.arange(w)).unsqueeze(0).expand(n, -1)
                lg = model(input_ids=ids.to(dev), attention_mask=att.to(dev),
                           position_ids=pos.to(dev),
                           past_key_values=self._fresh_cache(kv, n),
                           use_cache=False).logits
                lp = torch.log_softmax(lg.float(), dim=-1)
                for i, x in enumerate(chunk):
                    tot = head[x[0]].item()
                    for k in range(1, len(x)):
                        tot += lp[i, k - 1, x[k]].item()
                    out.append(tot / len(x))
        self.calls += len(conts)
        return out

    def score(self, jobs):
        """jobs is [(prompt, continuation)]; returns scores in order."""
        if self.ok is False:
            self.calls += len(jobs)
            return self.base.score(jobs)
        groups = defaultdict(list)
        for i, (p, c) in enumerate(jobs):
            groups[p].append(i)
        res = [None] * len(jobs)
        for p, idx in groups.items():
            conts = [jobs[i][1] for i in idx]
            if self.ok is None:
                try:
                    fast = self._group(p, conts)
                    slow = self.base.score([(p, c) for c in conts])
                    diff = max(abs(x - y) for x, y in zip(fast, slow))
                    self.ok = diff < 1e-3
                    print(f"  shared-prompt scoring self-check: max difference "
                          f"{diff:.2e} -> "
                          f"{'using it' if self.ok else 'disabled, plain scoring'}",
                          file=sys.stderr)
                    vals = slow
                except Exception as e:                    # noqa: BLE001
                    self.ok = False
                    print(f"  shared-prompt scoring unavailable "
                          f"({type(e).__name__}: {str(e)[:80]}); "
                          f"using plain scoring", file=sys.stderr)
                    vals = self.base.score([(p, c) for c in conts])
            elif self.ok:
                vals = self._group(p, conts)
            else:
                vals = self.base.score([(p, c) for c in conts])
            for i, v in zip(idx, vals):
                res[i] = v
        return res


class ResumableScorer:
    """Wraps a scorer so finished work survives a timed-out job.

    Every score is appended to a cache file as soon as its chunk finishes,
    keyed by a hash of prompt and continuation. A resubmitted job reads the
    file, skips what is done, and continues. Uncached jobs are scored
    shortest first, so each batch holds sequences of similar length and
    little compute is spent on padding.
    """

    def __init__(self, scorer, path, chunk=256):
        self.sc, self.path, self.chunk = scorer, path, chunk
        self.memo = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        k, v = json.loads(line)
                        self.memo[k] = v
                    except (ValueError, TypeError):
                        continue        # a half-written last line
            print(f"  resuming: {len(self.memo):,} scores already in {path}",
                  file=sys.stderr)

    @property
    def calls(self):
        return self.sc.calls

    @staticmethod
    def key(prompt, cont):
        return hashlib.sha1((prompt + "\x00" + cont).encode("utf-8")).hexdigest()

    def score(self, jobs):
        keys = [self.key(p, c) for p, c in jobs]
        uniq = set(keys)
        cached = sum(1 for k in uniq if k in self.memo)
        todo, seen = [], set()
        for i, k in enumerate(keys):
            if k not in self.memo and k not in seen:
                seen.add(k)
                todo.append(i)
        # same-prompt jobs adjacent, shortest prompts first
        todo.sort(key=lambda i: (len(jobs[i][0]), jobs[i][0], len(jobs[i][1])))
        print(f"  {len(jobs):,} scores needed, {len(uniq):,} distinct; "
              f"{cached:,} already in the cache, {len(todo):,} to compute",
              file=sys.stderr)
        fh = open(self.path, "a", encoding="utf-8") if self.path else None
        t0 = time.time()
        try:
            for b in range(0, len(todo), self.chunk):
                idx = todo[b:b + self.chunk]
                got = self.sc.score([jobs[i] for i in idx])
                for i, v in zip(idx, got):
                    self.memo[keys[i]] = v
                    if fh:
                        fh.write(json.dumps([keys[i], v]) + "\n")
                if fh:
                    fh.flush()
                    os.fsync(fh.fileno())
                done = b + len(idx)
                rate = done / max(1e-9, time.time() - t0)
                left = (len(todo) - done) / max(rate, 1e-9) / 60
                print(f"    {done:,}/{len(todo):,} scored, about {left:.0f} "
                      f"min left (longest prompts come last)",
                      file=sys.stderr, flush=True)
        finally:
            if fh:
                fh.close()
        return [self.memo[k] for k in keys]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", required=True,
                    help="an edge file with source, target, relation, tier")
    ap.add_argument("--wiki-cache", default="wiki_cache")
    ap.add_argument("--extra-text", nargs="*", default=[],
                    help="further corpus caches to read alongside the "
                         "article, such as pubmed_cache. The recovery step "
                         "has to see the same corpus the tier was measured "
                         "against: a tier taken from tier_union but searched "
                         "in Wikipedia alone loses every edge whose category "
                         "was named in the other corpus")
    ap.add_argument("--ancestors", required=True)
    ap.add_argument("--ancestor-cols", default="")
    ap.add_argument("--flip-ancestors", action="store_true")
    ap.add_argument("--names", choices=["redirects", "gbif", "none"],
                    default="redirects")
    ap.add_argument("--names-cache", default="gbif_cache")
    ap.add_argument("--expand-depth", type=int, default=2,
                    help="must match the --ancestor-depth the build used")
    ap.add_argument("--min-candidates", type=int, default=2,
                    help="an expansion of one is not a choice; those edges "
                         "are reported separately as solved symbolically")
    ap.add_argument("--max-candidates", type=int, default=64,
                    help="cap on the candidate set. The cap is applied to "
                         "the sorted expansion, so it can drop the target; "
                         "that is counted against `ours` rather than hidden")
    ap.add_argument("--ctx-chars", type=int, default=6000,
                    help="with --ctx-mode prefix, how much of the page to give")
    ap.add_argument("--raw-passage", action="store_true",
                    help="give the model Wikipedia source markup instead of "
                         "cleaned prose (the earlier behaviour)")
    ap.add_argument("--ctx-mode", choices=["snippet", "prefix"],
                    default="snippet",
                    help="snippet: the lead plus the passages around the "
                         "category mention, as a retriever would return them; "
                         "prefix: the first --ctx-chars characters")
    ap.add_argument("--no-shared-prompt", action="store_true",
                    help="score every (prompt, candidate) pair from scratch; "
                         "slower, and only needed if the self-check fails")
    ap.add_argument("--cache", default="",
                    help="score cache for resuming; defaults to <out>.cache.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if not os.path.exists(a.scored):
        sys.exit(f"missing: {a.scored}")
    if not os.path.isdir(a.wiki_cache):
        sys.exit(f"no article cache at {a.wiki_cache}")
    with open(a.scored, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{a.scored} has no rows")
    for need in ("source", "target", "relation", "tier"):
        if need not in rows[0]:
            sys.exit(f"{a.scored} has no '{need}' column. Present: "
                     + ", ".join(sorted(rows[0])))

    anc = nt.load_ancestors(a.ancestors, a.ancestor_cols, a.flip_ancestors)
    if not anc:
        sys.exit("the ancestor table is empty")
    children = defaultdict(set)
    for ch, ps in anc.items():
        for p in ps:
            children[p].add(ch)

    def names_for(e):
        if a.names == "gbif":
            n = nt.read_names(a.names_cache, "gbif", e)
            if not n:
                return nt.EMPTY
            out = []
            for rk in (n.get("ancestors") or [])[:a.expand_depth]:
                out.append({"rank": rk.get("rank", ""),
                            "scientific": rk.get("scientific", ""),
                            "vernacular": (rk.get("vernacular") or [])[:4]})
            return {"synonyms": n.get("synonyms", []),
                    "vernacular": n.get("vernacular", []), "ancestors": out}
        if a.names == "redirects":
            n = dict(nt.read_names(a.names_cache, "redir", e) or nt.EMPTY)
            n["synonyms"] = []
            n["ancestors"] = nt.climb(e, anc, a.expand_depth)
            return n
        return nt.EMPTY

    # Target pools and the true partners of each source, for the controls.
    pool = defaultdict(set)
    partners = defaultdict(set)
    for r in rows:
        pool[r["relation"]].add(r["target"])
        partners[(r["relation"], r["source"])].add(r["target"])
    pool = {k: sorted(v) for k, v in pool.items()}

    tier3 = [r for r in rows if str(r["tier"]).strip() == "3"]
    if a.limit:
        tier3 = tier3[:a.limit]
    print(f"\n{len(tier3):,} edges at tier 3 of {len(rows):,}")
    if not tier3:
        sys.exit("nothing at tier 3; check that the build ran with --ancestors")

    for d in a.extra_text:
        if not os.path.isdir(d):
            sys.exit(f"no corpus cache at {d}")

    def load_text(src):
        """The article plus every extra corpus document for this entity.

        The extra caches name their files with a different sanitizer and may
        carry a query-set suffix, so they are matched by prefix rather than
        by an exact path.
        """
        parts = []
        t = nt.read_article(a.wiki_cache, src)
        if t:
            parts.append(t)
        n_extra = 0
        for d in a.extra_text:
            for p in sorted(glob.glob(os.path.join(
                    d, nt._safe(src) + "*.txt"))):
                try:
                    with open(p, encoding="utf-8", errors="replace") as fh:
                        body = fh.read()
                except OSError:
                    continue
                if body.strip():
                    parts.append(body)
                    n_extra += 1
        return ("\n\n".join(parts) if parts else None), n_extra

    rng = random.Random(a.seed)
    items, no_art, no_cat, singleton = [], 0, 0, 0
    arts, with_extra = {}, 0
    t0 = time.time()
    for i, r in enumerate(tier3):
        src, tgt, rel = r["source"], r["target"], r["relation"]
        if src not in arts:
            arts[src] = load_text(src)
        text, n_extra = arts[src]
        if n_extra:
            with_extra += 1
        if text is None:
            no_art += 1
            continue
        ancs = [x["scientific"] for x in nt.climb(tgt, anc, a.expand_depth)]
        matched = [x for x in ancs
                   if x and not nt.too_generic(x) and nt.mentions(text, x)]
        if not matched:
            no_cat += 1
            continue
        # The narrowest matched category gives the smallest candidate set,
        # which is the one a method has any reason to pay for.
        best, bestset = None, None
        for x in matched:
            s = expand(x, children, a.expand_depth)
            if bestset is None or len(s) < len(bestset):
                best, bestset = x, s
        cands = sorted(bestset)
        capped = len(cands) > a.max_candidates
        cands = cands[:a.max_candidates]
        if len(cands) < a.min_candidates:
            singleton += 1
            continue
        items.append({
            "row": r, "src": src, "tgt": tgt, "rel": rel, "text": text,
            "category": best, "cands": cands, "k": len(cands),
            "capped": int(capped),
            "has_gold": int(tgt.lower() in {c.lower() for c in cands}),
            "n_matched": len(matched), "matched": matched,
            "lookups": len(ancs) + 1,
        })
        if (i + 1) % 200 == 0:
            print(f"    expanded {i + 1}/{len(tier3)}", end="\r",
                  file=sys.stderr)
    print(" " * 44, end="\r", file=sys.stderr)
    print(f"  expansion took {time.time() - t0:.0f}s")
    if a.extra_text:
        print(f"  {with_extra:,} edges also carry text from "
              f"{', '.join(a.extra_text)}")
    if no_art:
        print(f"  {no_art:,} sources have no cached text")
    if no_cat:
        print(f"  {no_cat:,} have no recoverable category name")
        if not a.extra_text:
            print("    If the tier column came from tier_union, these are "
                  "mostly edges whose\n    category was named in the other "
                  "corpus. Pass --extra-text <its cache>.")
    if singleton:
        print(f"  {singleton:,} expand to a single candidate, so the category "
              f"names the target outright\n    and no model call is needed; "
              f"they are excluded from the comparison")
    if not items:
        sys.exit("nothing usable")

    ks = [it["k"] for it in items]
    with_gold = [it for it in items if it["has_gold"]]
    print(f"\n  {len(items):,} edges enter the comparison")
    print(f"  candidates per edge: median "
          f"{sorted(ks)[len(ks) // 2]}, mean {sum(ks) / len(ks):.1f}, "
          f"max {max(ks)}")
    print(f"  mean chance rate 1/k = "
          f"{100 * sum(1 / k for k in ks) / len(ks):.2f}%")
    print(f"  expansion contains the target: {len(with_gold):,} "
          f"({100 * len(with_gold) / len(items):.2f}%)")
    if sum(it["capped"] for it in items):
        print(f"  the cap of {a.max_candidates} bites on "
              f"{sum(it['capped'] for it in items):,} edges, and is the only "
              f"reason an expansion\n    can miss the target at this depth")

    for it in items:
        if a.ctx_mode == "snippet":
            it["passage"] = snippet(it["text"] or "", it["matched"],
                                    clean=not a.raw_passage)
        else:
            raw = it["text"] or ""
            it["passage"] = (raw if a.raw_passage else clean_wikitext(raw))[:a.ctx_chars]
    shows = sum(1 for it in items
                if any(nt.mentions(it["passage"], m) for m in it["matched"]))
    print(f"  passages that show the category to the model: {shows:,} of "
          f"{len(items):,}")
    if shows < len(items):
        print("    the rest named it only inside markup or past the length "
              "cap; those edges\n    give the page-in-prompt conditions "
              "nothing to eliminate with")

    if a.dry_run:
        print("\nDry run, so no model was loaded. Everything above is "
              "symbolic and bounds\nwhat the conditions can reach.")
        return

    cache = a.cache or (a.out + ".cache.jsonl" if a.out else "")
    base = Scorer(a.model, a.threads, a.batch_size, a.gpu)
    sc = ResumableScorer(base if a.no_shared_prompt else PrefixScorer(base),
                         cache)

    # Option sets. Every condition gets exactly k options including the
    # target, so chance is 1/k in all four and the comparison is at matched
    # difficulty.
    for it in items:
        k = it["k"]
        ours = list(it["cands"])
        if not it["has_gold"]:
            ours = ours[:k]                 # the target is genuinely absent
        order = list(range(len(ours)))
        rng.shuffle(order)
        it["opts_ours"] = [ours[j] for j in order]
        it["gold_ours"] = next(
            (j for j, o in enumerate(it["opts_ours"])
             if o.lower() == it["tgt"].lower()), None)

        cand = pool[it["rel"]]
        bad = partners[(it["rel"], it["src"])]
        ctrl, guard = [it["tgt"]], 0
        while len(ctrl) < k and guard < 400:
            guard += 1
            c = rng.choice(cand)
            if c not in bad and c not in ctrl:
                ctrl.append(c)
        rng.shuffle(ctrl)
        it["opts_ctrl"] = ctrl
        it["gold_ctrl"] = ctrl.index(it["tgt"])

    jobs, jmap = [], []
    for it in items:
        base = stem(it["rel"], it["src"])
        art = it["passage"]
        pre = (art + "\n\n") if art else ""
        for cond, opts, prompt in (
                ("ours", it["opts_ours"], base),
                ("ours_ctx", it["opts_ours"], pre + base),
                ("random_k", it["opts_ctrl"], base),
                ("rag_pool", it["opts_ctrl"], pre + base)):
            for j, o in enumerate(opts):
                jobs.append((prompt, " " + o))
                jmap.append((it, cond, j))

    print(f"\n{len(jobs):,} forward passes", file=sys.stderr)
    t0 = time.time()
    got = sc.score(jobs)
    print(f"  {time.time() - t0:.0f}s", file=sys.stderr)

    acc = defaultdict(lambda: defaultdict(dict))
    for (it, cond, j), s in zip(jmap, got):
        acc[id(it)][cond][j] = s
    for it in items:
        d = acc[id(it)]
        for cond in CONDS:
            gold = (it["gold_ours"] if cond in ("ours", "ours_ctx")
                    else it["gold_ctrl"])
            if cond not in d or gold is None:
                it[f"hit_{cond}"] = 0
                it[f"pick_{cond}"] = ""
                continue
            pick = max(d[cond], key=lambda q: d[cond][q])
            opts = (it["opts_ours"] if cond in ("ours", "ours_ctx")
                    else it["opts_ctrl"])
            it[f"pick_{cond}"] = opts[pick]
            it[f"hit_{cond}"] = int(pick == gold)

    def block(title, sel):
        print(f"\n  {title}   n = {len(sel):,}")
        if not sel:
            return
        ch = 100 * sum(1 / it["k"] for it in sel) / len(sel)
        print(f"    chance 1/k = {ch:.2f}%")
        for cond in CONDS:
            k = sum(it[f"hit_{cond}"] for it in sel)
            p, lo, hi = wilson(k, len(sel))
            print(f"    {cond:<10} {p:6.2f}% [{lo:5.2f}-{hi:5.2f}]"
                  f"   chance-corrected {p - ch:+6.2f}")

    print(f"\n{'=' * 72}\nRECOVERY")
    block("ON RECOVERABLE EDGES", with_gold)
    block("OVERALL, expansion misses counted against ours", items)

    print("\n  by relation, on recoverable edges")
    by = defaultdict(list)
    for it in with_gold:
        by[it["rel"]].append(it)
    for rel in sorted(by):
        block(rel.upper(), by[rel])

    print("\n  by candidate-set size, on recoverable edges")
    for lo, hi, lab in ((2, 4, "k 2-4"), (5, 12, "k 5-12"),
                        (13, 10 ** 9, "k 13+")):
        sel = [it for it in with_gold if lo <= it["k"] <= hi]
        if len(sel) >= 15:
            block(lab, sel)

    print(f"\n{'=' * 72}\nPAIRED COMPARISONS, on recoverable edges")
    for aa, bb, note in (
            ("ours", "rag_pool", "the result: ontology expansion against "
                                 "naive retrieval at matched k"),
            ("ours", "random_k", "whether the expansion beats an arbitrary "
                                 "set of the same size"),
            ("ours_ctx", "ours", "whether the article adds anything once the "
                                 "category is extracted")):
        x = sum(1 for it in with_gold
                if it[f"hit_{aa}"] and not it[f"hit_{bb}"])
        y = sum(1 for it in with_gold
                if it[f"hit_{bb}"] and not it[f"hit_{aa}"])
        m = mcnemar(x, y)
        print(f"\n  {aa} against {bb}")
        print(f"    {note}")
        print(f"    {aa} only {x:,}   {bb} only {y:,}   "
              f"discordant {m['n_disc']:,}   chi2={m['chi']:.2f}  "
              f"p={m['p']:.3g}")
        if m["p"] >= 0.05:
            print(f"    not significant at n={len(with_gold):,}")

    print(f"\n{'=' * 72}\nCOST")
    n = len(items)
    print(f"  model forward passes per edge   {sc.calls / n:8.1f}")
    print(f"  ontology lookups per edge       "
          f"{sum(it['lookups'] for it in items) / n:8.1f}")
    print(f"  candidates scored per edge      {sum(ks) / n:8.1f}")

    if a.out:
        cols = (["source", "target", "relation", "category", "k", "capped",
                 "has_gold", "n_matched", "lookups"]
                + [f"pick_{c}" for c in CONDS] + [f"hit_{c}" for c in CONDS])
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for it in items:
                rec = {"source": it["src"], "target": it["tgt"],
                       "relation": it["rel"], "category": it["category"],
                       "k": it["k"], "capped": it["capped"],
                       "has_gold": it["has_gold"],
                       "n_matched": it["n_matched"],
                       "lookups": it["lookups"]}
                for c in CONDS:
                    rec[f"pick_{c}"] = it.get(f"pick_{c}", "")
                    rec[f"hit_{c}"] = it.get(f"hit_{c}", 0)
                w.writerow(rec)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
