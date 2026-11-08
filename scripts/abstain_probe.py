#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do models answer when the page does not say?

The question this answers is the one RAG systems are judged on: given a
passage that does not state the answer, does the model say so, or does it
answer anyway? Every edge is asked as a lettered multiple-choice question,
and one of the letters is always "Not enough information to tell". A model
can therefore get it right, get it wrong, or decline, and the three are
counted separately.

Edges are drawn in three groups by what the page says about the target:

  named      the page names the target (tiers 0-2)
  category   the page names only the target's parent category (tier 3)
  silent     the page names neither (tier 4)

and each question is asked twice, with the page and without it.

The distractors are, by default, the target's siblings in the ontology:
other members of the same category. That choice is what makes the category
group a real test. If the distractors were random diseases, a page saying
"diabetes" would let the model pick the one diabetes option by elimination,
and the category group would look easy for a reason that has nothing to do
with knowing the answer. With siblings as distractors, a page naming only
"diabetes" rules out nothing among the options, so the passage genuinely
does not determine the answer, and the only honest responses are to know it
from memory or to decline. The same siblings are used in all three groups,
so the groups differ only in what the page says.

The passage given to the model is the one a retriever centered on the
target would return: the opening of the article, plus the text around every
mention of the target, its other names, and its ontology ancestors. Silent
edges therefore get the opening alone, because nothing on the page is about
the target.

Scoring reads the model's next-token distribution over the option letters,
one forward pass per question, so the large model is affordable on CPU.

    # what would be asked, and how many siblings exist; no model
    python3 abstain_probe.py --scored me_bal_union.csv \\
        --wiki-cache wiki_cache --extra-text pubmed_cache \\
        --ancestors mondo_child_parent.csv --names-cache gbif_cache \\
        --dry-run

    python3 abstain_probe.py --scored me_bal_union.csv \\
        --wiki-cache wiki_cache --extra-text pubmed_cache \\
        --ancestors mondo_child_parent.csv --names-cache gbif_cache \\
        --model Qwen/Qwen2.5-7B-Instruct --threads 12 \\
        --out abstain_7B.csv

Resumable: every answered question is appended to <out>.cache.jsonl as it
finishes, and a rerun skips what is done.

Needs name_tiers.py, router.py and recover.py beside it, and torch plus
transformers unless --dry-run.
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import name_tiers as nt
    from recover import snippet
    from router import wilson
except ImportError as e:
    sys.exit(f"name_tiers.py, router.py and recover.py have to sit beside "
             f"this script: {e}")

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

BANDS = [("named", 0, 2), ("category", 3, 3), ("silent", 4, 4)]
NOTA = "Not enough information to tell"
# prompting baseline (--instruct): one added sentence, passage prompts only
INSTRUCT_LINE = ("\nIf the passage names only a broad category of diseases and "
                 "does not say which specific disease is meant, answer with the "
                 "letter of \"" + NOTA + "\".")
LETTERS = "ABCDEFGHIJ"

QUESTIONS = {
    "indication": "Which disease is {s} used to treat?",
    "contraindication": "In patients with which disease must {s} not be given?",
    "off-label use": "Which disease is {s} sometimes used off-label to treat?",
    "treats": "Which disease is {s} used to treat?",
    "palliates": "The symptoms of which disease does {s} relieve?",
    "eats": "Which species does {s} feed on?",
    "preysOn": "Which species does {s} prey on?",
    "parasiteOf": "Which species is {s} a parasite of?",
    "pollinates": "Which plant does {s} pollinate?",
    "visitsFlowersOf": "The flowers of which plant does {s} visit?",
    "hasHost": "Which species is the host of {s}?",
}


def question_for(rel, src):
    t = QUESTIONS.get(rel)
    if t is None:
        return f"Which of the following is related to {src} by the relation '{rel}'?"
    return t.format(s=src)


def two_prop(k1, n1, k2, n2):
    if not n1 or not n2:
        return None
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return 100 * (p1 - p2), z, p


# ----------------------------------------------------------------- the model

class LetterScorer:
    """Next-token log-probabilities over option letters, batched.

    Prompts are left-padded so the final position is real for every row,
    and position ids are rebuilt from the attention mask so the padding
    does not shift them. The batched scores match unbatched ones.
    """

    def __init__(self, model_name, threads=0, batch_size=4, gpu=False,
                 dtype="float32"):
        try:
            import torch
        except BaseException as e:                       # noqa: BLE001
            sys.exit(f"could not import torch: {type(e).__name__}: {e}\n"
                     f"This is the environment, not the CPU. Activate the "
                     f"CPU-only one:\n  source ~/cpu_probe_env/bin/activate")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
        self.bs = batch_size
        print(f"loading {model_name} on {self.device}, "
              f"{torch.get_num_threads()} threads ...", file=sys.stderr)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            m = AutoModelForCausalLM.from_pretrained(model_name,
                                                     dtype=getattr(torch, dtype))
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(model_name,
                                                     torch_dtype=getattr(torch, dtype))
        self.model = m.to(self.device).eval()
        self.chat = bool(getattr(self.tok, "chat_template", None))
        # The token a model emits for "B" may or may not carry a leading
        # space depending on the template, so both spellings are scored and
        # the better one is kept.
        self.letter_ids = {}
        for L in LETTERS:
            ids = set()
            for form in (L, " " + L):
                enc = self.tok(form, add_special_tokens=False)["input_ids"]
                if len(enc) == 1:
                    ids.add(enc[0])
            if not ids:
                sys.exit(f"the tokenizer has no single token for {L!r}")
            self.letter_ids[L] = sorted(ids)

    def render(self, user_text):
        if self.chat:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False, add_generation_prompt=True)
        return user_text + "\nAnswer:"

    def score(self, prompts, n_letters):
        """For each prompt, log-probs of letters A.. (renormalized over them)."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for b in range(0, len(prompts), self.bs):
                chunk = [self.render(p) for p in prompts[b:b + self.bs]]
                enc = [self.tok(c, add_special_tokens=False)["input_ids"]
                       for c in chunk]
                width = max(len(e) for e in enc)
                pad = self.tok.pad_token_id
                ids = torch.full((len(enc), width), pad, dtype=torch.long)
                att = torch.zeros((len(enc), width), dtype=torch.long)
                for i, e in enumerate(enc):
                    ids[i, width - len(e):] = torch.tensor(e, dtype=torch.long)
                    att[i, width - len(e):] = 1
                pos = (att.cumsum(-1) - 1).clamp(min=0)
                ids, att, pos = (t.to(self.device) for t in (ids, att, pos))
                logits = self.model(input_ids=ids, attention_mask=att,
                                    position_ids=pos).logits[:, -1, :]
                lp = torch.log_softmax(logits.float(), dim=-1)
                for i in range(len(enc)):
                    raw = []
                    for L in LETTERS[:n_letters[b + i]]:
                        raw.append(max(lp[i, t].item()
                                       for t in self.letter_ids[L]))
                    m_ = max(raw)
                    z = m_ + math.log(sum(math.exp(r - m_) for r in raw))
                    out.append([r - z for r in raw])
        return out


# ------------------------------------------------------------------ the data

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", required=True,
                    help="an edge file with source, target, relation, tier")
    ap.add_argument("--wiki-cache", default="wiki_cache")
    ap.add_argument("--extra-text", nargs="*", default=[],
                    help="further corpus caches; must match how the tier "
                         "column was measured (e.g. pubmed_cache for tier_union)")
    ap.add_argument("--ancestors", required=True)
    ap.add_argument("--ancestor-cols", default="")
    ap.add_argument("--flip-ancestors", action="store_true")
    ap.add_argument("--names", choices=["redirects", "gbif", "none"],
                    default="redirects")
    ap.add_argument("--names-cache", default="gbif_cache")
    ap.add_argument("--depth", type=int, default=2,
                    help="ancestor depth, as in the build")
    ap.add_argument("--per-band", type=int, default=150,
                    help="edges per group per relation")
    ap.add_argument("--options", type=int, default=5,
                    help="answer options, not counting the decline option")
    ap.add_argument("--distractors", choices=["sibling", "random"],
                    default="sibling")
    ap.add_argument("--conditions", default="passage,closed",
                    help="comma list from: passage, closed")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--instruct", action="store_true",
                    help="prompting baseline: tell the model to decline when the "
                         "passage names only a broad category (passage prompts only)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N questions, for a smoke test")
    ap.add_argument("--raw-passage", action="store_true",
                    help="give the model Wikipedia source markup instead of "
                         "cleaned prose (the earlier behaviour)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.options + 1 > len(LETTERS):
        sys.exit(f"at most {len(LETTERS) - 1} options")
    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    for c in conds:
        if c not in ("passage", "closed"):
            sys.exit(f"unknown condition {c!r}")
    if not os.path.exists(a.scored):
        sys.exit(f"missing: {a.scored}")
    if not os.path.isdir(a.wiki_cache):
        sys.exit(f"no article cache at {a.wiki_cache}")
    for d in a.extra_text:
        if not os.path.isdir(d):
            sys.exit(f"no corpus cache at {d}")

    with open(a.scored, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for need in ("source", "target", "relation", "tier"):
        if rows and need not in rows[0]:
            sys.exit(f"{a.scored} has no '{need}' column")

    anc = nt.load_ancestors(a.ancestors, a.ancestor_cols, a.flip_ancestors)
    children = defaultdict(set)
    for ch, ps in anc.items():
        for p in ps:
            children[p].add(ch)

    def names_for(e):
        if a.names == "gbif":
            n = nt.read_names(a.names_cache, "gbif", e) or {}
            out = [{"rank": rk.get("rank", ""),
                    "scientific": rk.get("scientific", ""),
                    "vernacular": (rk.get("vernacular") or [])[:4]}
                   for rk in (n.get("ancestors") or [])[:a.depth]]
            return {"synonyms": n.get("synonyms", []),
                    "vernacular": n.get("vernacular", []), "ancestors": out}
        if a.names == "redirects":
            n = dict(nt.read_names(a.names_cache, "redir", e) or nt.EMPTY)
            n["synonyms"] = []
            n["ancestors"] = nt.climb(e, anc, a.depth)
            return n
        return nt.EMPTY

    texts = {}

    def load_text(src):
        if src in texts:
            return texts[src]
        parts = []
        t = nt.read_article(a.wiki_cache, src)
        if t:
            parts.append(t)
        for d in a.extra_text:
            for p in sorted(glob.glob(os.path.join(d, nt._safe(src) + "*.txt"))):
                with open(p, encoding="utf-8", errors="replace") as fh:
                    body = fh.read()
                if body.strip():
                    parts.append(body)
        texts[src] = "\n\n".join(parts) if parts else None
        return texts[src]

    pool = defaultdict(set)
    partners = defaultdict(set)
    for r in rows:
        pool[r["relation"]].add(r["target"])
        partners[(r["relation"], r["source"])].add(r["target"])
    pool = {k: sorted(v) for k, v in pool.items()}

    # ---------------------------------------------------- sample by group
    rng = random.Random(a.seed)
    by = defaultdict(list)
    for r in rows:
        try:
            t = int(r["tier"])
        except (TypeError, ValueError):
            continue
        band = next((n for n, lo, hi in BANDS if lo <= t <= hi), None)
        if band:
            by[(r["relation"], band)].append(r)
    picked = []
    for key in sorted(by):
        v = list(by[key])
        rng.shuffle(v)
        picked.extend(v)            # quota is applied while building, so a
                                    # dropped edge does not shrink its group

    # ------------------------------------------------------- build items
    items, no_text, short_pool = [], 0, 0
    taken = defaultdict(int)
    hidden = defaultdict(int)
    sib_counts = defaultdict(list)
    for r in picked:
        src, tgt, rel = r["source"], r["target"], r["relation"]
        band = next(n for n, lo, hi in BANDS if lo <= int(r["tier"]) <= hi)
        if taken[(rel, band)] >= a.per_band:
            continue
        text = load_text(src)
        if text is None:
            no_text += 1
            continue
        # A distractor must be wrong. The target's ancestors are not: a drug
        # that treats allergic asthma treats asthma. Nor are its
        # descendants, or any true partner of the source.
        lineage = {x["scientific"].lower() for x in nt.climb(tgt, anc, 12)}
        frontier, seen_d = [tgt], set()
        while frontier:
            nxt = []
            for n_ in frontier:
                for c in children.get(n_, ()):
                    if c.lower() not in seen_d:
                        seen_d.add(c.lower())
                        nxt.append(c)
            frontier = nxt
        lineage |= seen_d
        bad = {x.lower() for x in partners[(rel, src)]} | {tgt.lower()} \
            | lineage | {src.lower()}
        dis = []
        if a.distractors == "sibling":
            # siblings first, then cousins one level further up
            for depth_up in (1, 2):
                ups = [x["scientific"] for x in nt.climb(tgt, anc, depth_up)]
                for up in ups:
                    for c in sorted(children.get(up, ())):
                        if c.lower() not in bad and c not in dis:
                            dis.append(c)
                if len(dis) >= a.options - 1:
                    break
            rng.shuffle(dis)
            dis = dis[:a.options - 1]
        n_sib = len(dis)
        guard = 0
        while len(dis) < a.options - 1 and guard < 500:
            guard += 1
            c = rng.choice(pool[rel])
            if c.lower() not in bad and c not in dis:
                dis.append(c)
        if len(dis) < a.options - 1:
            short_pool += 1
            continue
        sib_counts[band].append(n_sib)
        opts = [tgt] + dis
        rng.shuffle(opts)
        nota_at = rng.randrange(len(opts) + 1)
        opts.insert(nota_at, NOTA)
        sets = nt.tier_sets(tgt, names_for(tgt))
        passage = snippet(text, sets[3], clean=not a.raw_passage)
        # The group is defined by what the page says; the model sees the
        # passage. Keep the two in agreement: a "named" question must show
        # the target in its passage, a "family" question must show the
        # family. One whose mention lived only in markup, or past the
        # length cap, is skipped and another edge drawn in its place.
        own = {x.lower() for x in sets[2]}
        fam = [x for x in sets[3] if x.lower() not in own]
        if band == "named" and not any(nt.mentions(passage, x) for x in sets[2]):
            hidden[band] += 1
            continue
        if band == "category" and not any(nt.mentions(passage, x) for x in fam):
            hidden[band] += 1
            continue
        taken[(rel, band)] += 1
        items.append({
            "id": f"{rel}|{band}|{src}|{tgt}", "rel": rel, "band": band,
            "tier": int(r["tier"]), "src": src, "tgt": tgt,
            "opts": opts, "gold": opts.index(tgt), "nota": nota_at,
            "n_sibling": n_sib,
            "passage": passage,
        })
    if a.limit:
        items = items[:a.limit]

    print(f"\n{len(items):,} questions "
          f"({a.options} options plus '{NOTA}')")
    if no_text:
        print(f"  {no_text:,} edges dropped: no cached text for the source")
    if short_pool:
        print(f"  {short_pool:,} edges dropped: too few distractors")
    for b_, n_ in hidden.items():
        print(f"  {n_:,} {b_} edges skipped: the cleaned passage no longer "
              f"shows the {'target' if b_ == 'named' else 'family'} "
              f"(replaced by other edges where available)")
    print("\n  group      relation               questions   all-sibling options")
    cnt = defaultdict(int)
    for it in items:
        cnt[(it["band"], it["rel"])] += 1
    for band, _lo, _hi in BANDS:
        for (b2, rel), n in sorted(cnt.items()):
            if b2 != band:
                continue
            full = sum(1 for it in items if it["band"] == band and
                       it["rel"] == rel and it["n_sibling"] >= a.options - 1)
            print(f"  {band:<10} {rel:<22} {n:>9,}   {100 * full / n:6.1f}%")
    if a.distractors == "sibling":
        print("\n  'all-sibling' is the share of questions whose distractors "
              "are all from the\n  target's own category. Where siblings run "
              "out, random targets fill in, which\n  makes those questions "
              "easier by elimination.")

    if a.dry_run:
        for band, _lo, _hi in BANDS:
            ex = next((it for it in items if it["band"] == band), None)
            if not ex:
                continue
            print(f"\n----- example, {band} -----")
            print(f"Passage ({len(ex['passage'])} chars):")
            print("  " + ex["passage"][:500].replace("\n", "\n  ")
                  + (" ..." if len(ex["passage"]) > 500 else ""))
            print(f"Question: {question_for(ex['rel'], ex['src'])}")
            for i, o in enumerate(ex["opts"]):
                mark = "  <- true" if i == ex["gold"] else (
                    "  <- decline" if i == ex["nota"] else "")
                print(f"  {LETTERS[i]}. {o}{mark}")
        print("\nDry run, so no model was loaded.")
        return

    # ------------------------------------------------------------ score
    cache = (a.out + ".cache.jsonl") if a.out else ""
    done = {}
    if cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            for line in fh:
                try:
                    k, v = json.loads(line)
                    done[k] = v
                except (ValueError, TypeError):
                    continue
        print(f"  resuming: {len(done):,} answers already in {cache}",
              file=sys.stderr)

    def prompt_for(it, cond):
        q = question_for(it["rel"], it["src"])
        lines = [f"{LETTERS[i]}. {o}" for i, o in enumerate(it["opts"])]
        body = (f"Question: {q}\nOptions:\n" + "\n".join(lines) +
                "\nAnswer with the letter of one option only.")
        if cond == "passage":
            if a.instruct:
                body += INSTRUCT_LINE
            return f"Passage:\n{it['passage']}\n\n{body}"
        return body

    def key_for(it, c):
        # the prompt's fingerprint is part of the key, so answers cached for
        # a different passage are never reused for this one
        h = hashlib.sha1(prompt_for(it, c).encode("utf-8")).hexdigest()[:12]
        return f"{it['id']}||{c}||{h}"

    todo = [(it, c) for it in items for c in conds
            if key_for(it, c) not in done]
    todo.sort(key=lambda x: len(prompt_for(*x)))
    print(f"\n{len(items) * len(conds) - len(todo):,} answered, "
          f"{len(todo):,} to go", file=sys.stderr)

    if todo:
        sc = LetterScorer(a.model, a.threads, a.batch_size, a.gpu, a.dtype)
        fh = open(cache, "a", encoding="utf-8") if cache else None
        t0, step = time.time(), max(8, a.batch_size * 4)
        try:
            for b in range(0, len(todo), step):
                chunk = todo[b:b + step]
                got = sc.score([prompt_for(it, c) for it, c in chunk],
                               [len(it["opts"]) for it, _c in chunk])
                for (it, c), lp in zip(chunk, got):
                    k = key_for(it, c)
                    done[k] = lp
                    if fh:
                        fh.write(json.dumps([k, lp]) + "\n")
                if fh:
                    fh.flush()
                    os.fsync(fh.fileno())
                n = b + len(chunk)
                rate = n / max(1e-9, time.time() - t0)
                print(f"    {n:,}/{len(todo):,}, about "
                      f"{(len(todo) - n) / max(rate, 1e-9) / 60:.0f} min left",
                      file=sys.stderr, flush=True)
        finally:
            if fh:
                fh.close()

    # ----------------------------------------------------------- report
    res = []
    for it in items:
        for c in conds:
            lp = done.get(key_for(it, c))
            if lp is None:
                continue
            pick = max(range(len(lp)), key=lambda i: lp[i])
            outcome = ("correct" if pick == it["gold"] else
                       "abstain" if pick == it["nota"] else "wrong")
            res.append({"id": it["id"], "relation": it["rel"],
                        "group": it["band"], "tier": it["tier"],
                        "source": it["src"], "target": it["tgt"],
                        "condition": c, "n_options": len(it["opts"]),
                        "n_sibling": it["n_sibling"],
                        "pick": it["opts"][pick], "outcome": outcome,
                        "p_pick": round(math.exp(lp[pick]), 4),
                        "p_decline": round(math.exp(lp[it["nota"]]), 4),
                        "p_true": round(math.exp(lp[it["gold"]]), 4)})

    def tally(sel):
        n = len(sel)
        k = {o: sum(1 for r in sel if r["outcome"] == o)
             for o in ("correct", "wrong", "abstain")}
        return n, k

    print(f"\n{'=' * 74}\nWHAT THE MODEL DOES   chance of a correct pick "
          f"is {100 / (a.options + 1):.1f}%")
    for c in conds:
        print(f"\n  {c.upper()}")
        print("    group        n    correct          wrong            declines")
        for band, _lo, _hi in BANDS:
            n, k = tally([r for r in res if r["condition"] == c and
                          r["group"] == band])
            if not n:
                continue
            cells = []
            for o in ("correct", "wrong", "abstain"):
                p, lo, hi = wilson(k[o], n)
                cells.append(f"{p:5.1f}% [{lo:4.1f}-{hi:4.1f}]")
            print(f"    {band:<9} {n:>5,}   " + "   ".join(cells))

    if "passage" in conds:
        print(f"\n{'=' * 74}\nWITH THE PAGE IN HAND, DOES IT ANSWER ANYWAY?")
        sel = {b: tally([r for r in res if r["condition"] == "passage" and
                         r["group"] == b]) for b, _l, _h in BANDS}
        for b in ("category", "silent"):
            n, k = sel[b]
            if n:
                print(f"  {b:<9} answers {100 * (n - k['abstain']) / n:5.1f}% "
                      f"of the time; {100 * k['wrong'] / n:5.1f}% are wrong")
        for (b1, b2) in (("category", "silent"), ("named", "category")):
            n1, k1 = sel[b1]
            n2, k2 = sel[b2]
            t = two_prop(k1["wrong"], n1, k2["wrong"], n2)
            if t:
                print(f"  wrong-answer rate, {b1} minus {b2}: {t[0]:+6.1f} "
                      f"points  z={t[1]:+.2f}  p={t[2]:.3g}")
        print("\n  The category group is the one the paper is about: the page "
              "says something\n  true and relevant, but not enough to pick "
              "among the options. A wrong answer\n  there is a confident "
              "guess, not a misreading.")

    if "passage" in conds and "closed" in conds:
        print(f"\n{'=' * 74}\nWHAT THE PAGE CHANGES, per group")
        idx = defaultdict(dict)
        for r in res:
            idx[r["id"]][r["condition"]] = r["outcome"]
        for band, _lo, _hi in BANDS:
            pairs = [v for k_, v in idx.items()
                     if f"|{band}|" in k_ and len(v) == 2]
            if not pairs:
                continue
            fixed = sum(1 for v in pairs if v["closed"] != "correct" and
                        v["passage"] == "correct")
            broke = sum(1 for v in pairs if v["closed"] == "correct" and
                        v["passage"] != "correct")
            newly_wrong = sum(1 for v in pairs if v["closed"] != "wrong" and
                              v["passage"] == "wrong")
            print(f"  {band:<9} n={len(pairs):>5,}   page fixes {fixed:>4,}   "
                  f"page breaks {broke:>4,}   page induces a new wrong "
                  f"answer {newly_wrong:>4,}")

    print(f"\n  by relation, with the page")
    for rel in sorted({r["relation"] for r in res}):
        for band, _lo, _hi in BANDS:
            n, k = tally([r for r in res if r["relation"] == rel and
                          r["group"] == band and r["condition"] == "passage"])
            if n:
                print(f"    {rel:<20} {band:<9} n={n:>4}  correct "
                      f"{100 * k['correct'] / n:5.1f}%  wrong "
                      f"{100 * k['wrong'] / n:5.1f}%  declines "
                      f"{100 * k['abstain'] / n:5.1f}%")

    if a.out and res:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
            w.writeheader()
            w.writerows(res)
        print(f"\nwrote {a.out}: {len(res):,} rows")


if __name__ == "__main__":
    main()
