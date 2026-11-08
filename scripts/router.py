#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5, second half. The three-branch router, against three baselines.

The measurement half of the paper says a two-hop question can fail for three
different reasons, and that a system which treats them alike cannot do well
on any of them. This is the method that acts on that. It reads the first-hop
tier at run time and takes one of three branches.

  tier 0-2   The article names the bridge. Find it in the text and answer
             the second hop from it. This is retrieval working as intended.

  tier 3     The article names only the bridge's category. Read the category
             off the page, ask the ontology for its members, score them, and
             answer the second hop from the one that wins. The text supplies
             the category and the ontology supplies the candidates; neither
             half can do this alone, which is what makes the branch the
             contribution rather than a heuristic.

  tier 4     The article says nothing about the bridge. No amount of reading
             reaches this item, so the branch abstains. Both readings are
             reported: abstention scored as a miss, and a fallback to
             closed-book, since a reader will want to see what refusing
             costs and what guessing costs.

Three baselines bound it. Closed-book asks the chained question with no
text. Naive retrieval prepends the article and asks the same question, which
is what a retrieval-augmented system does today. The oracle is handed the
true bridge and answers only the second hop, which is the ceiling: every
condition uses the identical second-hop stem, and the only thing that
differs between the oracle and the router is which entity fills the subject
slot. The gap between them is therefore entirely a bridge-identification
gap, with nothing else mixed into it.

The honest headline is not overall accuracy. The router can only beat naive
retrieval on tier-3 items, which are a tenth of a graph, and it cannot beat
anything on tier-4 items, where its contribution is declining to answer. The
report is laid out to make both of those visible rather than buried in an
average.

    # symbolic parts only: branches, candidate sets, the ceiling. No torch.
    python3 router.py --items twohop_pkg.csv --wiki-cache wiki_cache \\
        --ancestors mondo_child_parent.csv --names-cache gbif_cache \\
        --dry-run

    python3 router.py --items twohop_pkg.csv --wiki-cache wiki_cache \\
        --ancestors mondo_child_parent.csv --names-cache gbif_cache \\
        --model Qwen/Qwen2.5-7B-Instruct --threads 12 --batch-size 8 \\
        --out router_pkg_7B.csv

Needs name_tiers.py beside it, and torch plus transformers unless --dry-run.
"""

import argparse
import csv
import math
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import name_tiers as nt
except ImportError as e:
    sys.exit(f"name_tiers.py has to sit beside this script: {e}")

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

# One stem per relation, subject in {s}. Every condition fills the same
# second-hop stem, so the comparison between them is a comparison of
# subjects and nothing else.
STEMS = {
    "indication": "{s} is a drug used to treat the disease called",
    "contraindication": "{s} is a drug that must not be given to patients who have",
    "off-label use": "{s} is a drug sometimes used off-label to treat",
    "treats": "{s} is a drug used to treat the disease called",
    "palliates": "{s} is a drug used to relieve the symptoms of",
    "disease_protein": "{s} is a disease associated with the gene",
    "disease_phenotype_positive": "{s} is a disease whose symptoms include",
    "disease_disease": "{s} is a disease related to the disease",
    "drug_protein": "{s} is a drug that binds the protein",
    "eats": "The species {s} feeds on the species",
    "preysOn": "The species {s} preys on the species",
    "parasiteOf": "The species {s} is a parasite of the species",
    "pollinates": "The species {s} pollinates the plant",
    "visitsFlowersOf": "The species {s} visits the flowers of",
    "hasHost": "The species {s} has as its host the species",
}
# The noun for a relation's target type, used to refer back to the bridge in
# the closed-book chain without naming it.
TYPES = {
    "indication": "disease", "contraindication": "disease",
    "off-label use": "disease", "treats": "disease", "palliates": "disease",
    "disease_protein": "gene", "disease_phenotype_positive": "symptom",
    "disease_disease": "disease", "drug_protein": "protein",
    "eats": "species", "preysOn": "species", "parasiteOf": "species",
    "pollinates": "plant", "visitsFlowersOf": "plant", "hasHost": "species",
}


def stem(rel, subj):
    t = STEMS.get(rel)
    if t is None:
        return f"{subj} {rel.replace('_', ' ')}"
    return t.format(s=subj)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h))


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mcnemar(a_only, b_only):
    """Paired test for two conditions on the same items.

    The unpaired two-proportion test is wrong here: the conditions answer
    the identical questions, so their errors are correlated and the unpaired
    standard error is too large. Only the discordant pairs carry
    information, which is what this counts.
    """
    n = a_only + b_only
    if n == 0:
        return {"n_disc": 0, "chi": 0.0, "p": 1.0}
    chi = (abs(a_only - b_only) - 1) ** 2 / n
    return {"n_disc": n, "chi": chi,
            "p": 2.0 * (1.0 - norm_cdf(math.sqrt(max(0.0, chi))))}


# ------------------------------------------------------------------ scoring

class Scorer:
    """Mean log-probability of each continuation under one prompt.

    Mean rather than sum, so a long entity name is not penalized for its
    length. Every condition is scored the same way, so whatever bias the
    choice carries is constant across the comparison.
    """

    def __init__(self, model_name, threads=0, batch_size=8, gpu=False):
        try:
            import torch
        except BaseException as e:                           # noqa: BLE001
            sys.exit(f"could not import torch: {type(e).__name__}: {e}\n"
                     f"See the note in run_probe_cpu.py about a CPU-only "
                     f"environment.")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        print(f"loading {model_name} on {self.device}, "
              f"{torch.get_num_threads()} threads ...", file=sys.stderr)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            m = AutoModelForCausalLM.from_pretrained(model_name,
                                                     dtype=torch.float32)
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(model_name,
                                                     torch_dtype=torch.float32)
        self.model = m.to(self.device).eval()
        self.calls = 0

    def score(self, jobs):
        """jobs is [(prompt, continuation)]. Returns a list of mean log-probs."""
        torch = self.torch
        out = [0.0] * len(jobs)
        self.calls += len(jobs)
        with torch.no_grad():
            for b in range(0, len(jobs), self.batch_size):
                chunk = jobs[b:b + self.batch_size]
                seqs, spans = [], []
                for prompt, cont in chunk:
                    p_ids = self.tok(prompt, add_special_tokens=True)["input_ids"]
                    c_ids = self.tok(cont, add_special_tokens=False)["input_ids"]
                    if not c_ids:
                        c_ids = [self.tok.eos_token_id]
                    seqs.append(p_ids + c_ids)
                    spans.append((len(p_ids), len(c_ids)))
                width = max(len(s) for s in seqs)
                pad = self.tok.pad_token_id or 0
                ids = torch.full((len(seqs), width), pad, dtype=torch.long)
                att = torch.zeros((len(seqs), width), dtype=torch.long)
                for i, s in enumerate(seqs):
                    ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
                    att[i, :len(s)] = 1
                ids, att = ids.to(self.device), att.to(self.device)
                logits = self.model(input_ids=ids, attention_mask=att).logits
                logp = torch.log_softmax(logits.float(), dim=-1)
                for i, (p_len, c_len) in enumerate(spans):
                    tot = 0.0
                    for k in range(c_len):
                        pos = p_len + k - 1
                        tgt = ids[i, p_len + k]
                        tot += logp[i, pos, tgt].item()
                    out[b + i] = tot / max(1, c_len)
        return out


# --------------------------------------------------------------- the router

def build_candidates(item, text, anc, children, names_for, depth, cap):
    """What the router may consider as the bridge, and how it got there.

    Returns (branch, candidates, n_lookups). The branch is decided by the
    tier of the first hop, which is a measurement, not a guess by the model.
    """
    tier = item["_tier"]
    if tier <= 2:
        # The article names the bridge. Which of the plausible bridges does
        # it name? This is a string match against the same name sets the
        # measurement used, so it is symbolic and costs no model calls.
        hits = []
        for cand in item["_bridge_pool"]:
            sets = names_for(cand)
            s2 = nt.tier_sets(cand, sets)[2]
            if any(nt.mentions(text, p) for p in s2):
                hits.append(cand)
        if not hits:
            # The measurement says the bridge is named, so an empty result
            # here means the pool does not contain it. Fall through rather
            # than claim a branch that has nothing to work with.
            return "1-empty", [], len(item["_bridge_pool"])
        return "1", hits[:cap], len(item["_bridge_pool"])

    if tier == 3:
        # The article names an ancestor. Recover which one, take the
        # narrowest, and list its members.
        tgt = item["bridge"]
        ancs = [x["scientific"] for x in nt.climb(tgt, anc, depth=depth)]
        matched = [x for x in ancs
                   if x and not nt.too_generic(x) and nt.mentions(text, x)]
        if not matched:
            return "2-empty", [], len(ancs)
        best, bestset = None, None
        for x in matched:
            s = expand(x, children, depth)
            if bestset is None or len(s) < len(bestset):
                best, bestset = x, s
        cands = sorted(bestset)
        over = len(cands) > cap
        return ("2-capped" if over else "2"), cands[:cap], len(ancs) + 1

    return "3", [], 0


def expand(name, children, depth):
    out, frontier, seen = set(), [name], {name.lower()}
    for _ in range(depth):
        nxt = []
        for n in frontier:
            for c in children.get(n, ()):
                if c.lower() in seen:
                    continue
                seen.add(c.lower())
                out.add(c)
                nxt.append(c)
        frontier = nxt
        if not frontier:
            break
    return out


# ----------------------------------------------------------------- reporting

def acc_block(title, rows, conds):
    print(f"\n  {title}   n = {len(rows):,}")
    if not rows:
        return
    for c in conds:
        got = [r for r in rows if r.get(f"ans_{c}") is not None]
        k = sum(1 for r in got if r[f"hit_{c}"])
        p, lo, hi = wilson(k, len(got))
        cov = 100 * len(got) / len(rows)
        oa = ""
        if len(got) < len(rows):
            oa = f"   answered {cov:5.1f}%"
        print(f"    {c:<16} {p:6.2f}% [{lo:5.2f}-{hi:5.2f}]  "
              f"n={len(got):>5,}{oa}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True, help="output of make_twohop.py")
    ap.add_argument("--wiki-cache", default="wiki_cache")
    ap.add_argument("--ancestors", required=True)
    ap.add_argument("--ancestor-cols", default="")
    ap.add_argument("--flip-ancestors", action="store_true")
    ap.add_argument("--names", choices=["redirects", "gbif", "none"],
                    default="redirects")
    ap.add_argument("--names-cache", default="gbif_cache")
    ap.add_argument("--expand-depth", type=int, default=2,
                    help="must match the --ancestor-depth the build used")
    ap.add_argument("--max-candidates", type=int, default=24,
                    help="cap on the bridge candidate set; the share of "
                         "items it bites is reported")
    ap.add_argument("--choices", type=int, default=5,
                    help="options for the second hop")
    ap.add_argument("--ctx-chars", type=int, default=6000,
                    help="article prefix given to the naive retrieval baseline")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="branches, candidate sets and the ceiling only; "
                         "no model is loaded")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if not os.path.exists(a.items):
        sys.exit(f"missing: {a.items}\nBuild it with make_twohop.py first.")
    if not os.path.isdir(a.wiki_cache):
        sys.exit(f"no article cache at {a.wiki_cache}; the router reads the "
                 f"same cache the build used")
    with open(a.items, encoding="utf-8") as fh:
        items = list(csv.DictReader(fh))
    if not items:
        sys.exit(f"{a.items} has no rows")
    for need in ("a", "r1", "bridge", "r2", "answer", "tier"):
        if need not in items[0]:
            sys.exit(f"{a.items} has no '{need}' column. Present: "
                     + ", ".join(sorted(items[0])))

    anc = nt.load_ancestors(a.ancestors, a.ancestor_cols, a.flip_ancestors)
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

    # Items with no measured tier cannot be routed, since the branch is
    # chosen by the measurement. They are dropped here and counted, rather
    # than assigned a tier by default, which would put unmeasured items into
    # whichever branch the default names.
    unmeasured = [it for it in items if str(it["tier"]).strip() == ""]
    items = [it for it in items if str(it["tier"]).strip() != ""]
    for it in items:
        it["_tier"] = int(it["tier"])
    # Several bridges can lead to the same answer, and the router succeeds
    # by finding any of them. Scoring a pick against one arbitrary member
    # would charge it for choosing a different correct path.
    for it in items:
        gold = (it.get("gold_bridges") or it["bridge"]).split("|")
        it["_gold_bridges"] = {g.strip().lower() for g in gold if g.strip()}
    if unmeasured:
        print(f"{len(unmeasured):,} items have no measured first-hop tier "
              f"and are dropped")
    if a.limit:
        items = items[:a.limit]
    if not items:
        sys.exit("no items carry a measured tier; rerun make_twohop.py "
                 "with --tiers")

    # Pools. Bridge candidates for branch 1 come from the bridges this file
    # contains, which is the set a system would have from the graph's own
    # node list. Answer options for the second hop come from the same
    # relation, so the choice is between entities of the right type.
    rng = random.Random(a.seed)
    bridges_by_rel = defaultdict(set)
    answers_by_rel = defaultdict(set)
    true_answers = defaultdict(set)
    for it in items:
        bridges_by_rel[it["r1"]].add(it["bridge"])
        answers_by_rel[it["r2"]].add(it["answer"])
        # Every answer reachable through any of this item's bridges is also
        # correct, so none of them may serve as a distractor. make_twohop.py
        # records the whole set; without it an item's other correct answers
        # would be offered as wrong ones and the accuracy would be
        # understated by roughly the fan-out of the second hop.
        valid = {it["answer"]}
        if it.get("valid_answers"):
            valid |= {v.strip() for v in it["valid_answers"].split("|")
                      if v.strip()}
        it["_valid"] = valid
        true_answers[(it["r2"], it["bridge"])] |= valid
    for it in items:
        it["_bridge_pool"] = sorted(bridges_by_rel[it["r1"]])

    print(f"\n{len(items):,} items, "
          f"{sum(len(v) for v in bridges_by_rel.values()):,} distinct bridges, "
          f"{sum(len(v) for v in answers_by_rel.values()):,} distinct answers")

    # Branch assignment and candidate sets. All symbolic.
    missing_art = 0
    t0 = time.time()
    for i, it in enumerate(items):
        text = nt.read_article(a.wiki_cache, it["a"])
        if text is None:
            missing_art += 1
            text = ""
        it["_text"] = text
        br, cands, lookups = build_candidates(
            it, text, anc, children, names_for, a.expand_depth,
            a.max_candidates)
        it["_branch"] = br
        it["_cands"] = cands
        it["_lookups"] = lookups
        it["_cand_has_gold"] = int(
            bool(it["_gold_bridges"] & {c.lower() for c in cands}))
        if (i + 1) % 200 == 0:
            print(f"    routed {i + 1}/{len(items)}", end="\r", file=sys.stderr)
    print(" " * 40, end="\r", file=sys.stderr)
    if missing_art:
        print(f"{missing_art:,} first entities have no cached article; their "
              f"text is treated as empty")
    print(f"routing took {time.time() - t0:.0f}s")

    print("\nBRANCHES")
    by_br = defaultdict(list)
    for it in items:
        by_br[it["_branch"]].append(it)
    print("  branch      n      share   candidates (mean)   gold in candidates")
    for br in sorted(by_br):
        v = by_br[br]
        cs = [len(x["_cands"]) for x in v]
        g = 100 * sum(x["_cand_has_gold"] for x in v) / len(v)
        print(f"  {br:<10} {len(v):>6,}  {100 * len(v) / len(items):5.1f}%  "
              f"{(sum(cs) / len(cs)) if cs else 0:14.1f}      {g:6.2f}%")
    print("\n  The last column is the ceiling on each branch. A branch cannot")
    print("  answer an item correctly more often than it puts the true bridge")
    print("  in front of the model, except by luck on the second hop.")

    n2 = len(by_br.get("2", [])) + len(by_br.get("2-capped", []))
    if by_br.get("2-capped"):
        print(f"\n  the candidate cap of {a.max_candidates} bites on "
              f"{len(by_br['2-capped']):,} of {n2:,} middle-branch items")
    if by_br.get("1-empty") or by_br.get("2-empty"):
        print(f"  {len(by_br.get('1-empty', [])):,} items are measured as "
              f"naming the bridge but no pooled name matches, and")
        print(f"  {len(by_br.get('2-empty', [])):,} are measured at tier 3 "
              f"but no ancestor name is recoverable.")
        print("  Both are counted as router failures rather than excluded.")

    if a.dry_run:
        print("\nDry run, so no model was loaded. The branch shares and the")
        print("gold-in-candidates column above are the parts that do not")
        print("depend on a model, and they bound everything that follows.")
        return

    # ------------------------------------------------------------- scoring
    sc = Scorer(a.model, a.threads, a.batch_size, a.gpu)

    def answer_options(it):
        pool = sorted(answers_by_rel[it["r2"]])
        bad = it["_valid"] | true_answers[(it["r2"], it["bridge"])]
        opts, guard = [it["answer"]], 0
        while len(opts) < a.choices and guard < 300:
            guard += 1
            c = rng.choice(pool)
            if c not in bad and c not in opts:
                opts.append(c)
        order = list(range(len(opts)))
        rng.shuffle(order)
        return [opts[i] for i in order], order.index(0)

    for it in items:
        it["_opts"], it["_gold"] = answer_options(it)

    def hop2(subject, it):
        return stem(it["r2"], subject)

    def chain_prompt(it):
        t1 = TYPES.get(it["r1"], "entity")
        return (f"{stem(it['r1'], it['a'])} a certain {t1}. "
                f"{hop2('that ' + t1, it)}")

    # Step one: pick a bridge, where a branch has candidates to pick from.
    pick_jobs, pick_map = [], []
    for it in items:
        cands = it["_cands"]
        it["_picked"] = ""
        if len(cands) == 1:
            it["_picked"] = cands[0]
        elif len(cands) > 1:
            ctx = ""
            if it["_branch"].startswith("2"):
                ctx = ""        # the page has already supplied the category
            base = ctx + stem(it["r1"], it["a"])
            for c in cands:
                pick_jobs.append((base, " " + c))
                pick_map.append((it, c))
    if pick_jobs:
        print(f"\nbridge selection: {len(pick_jobs):,} forward passes",
              file=sys.stderr)
        got = sc.score(pick_jobs)
        best = {}
        for (it, c), s in zip(pick_map, got):
            key = id(it)
            if key not in best or s > best[key][0]:
                best[key] = (s, c)
        for it in items:
            if id(it) in best:
                it["_picked"] = best[id(it)][1]
    for it in items:
        it["_bridge_right"] = (
            int(it["_picked"].lower() in it["_gold_bridges"])
            if it["_picked"] else 0)

    # Step two: the four conditions, all over the same answer options.
    conds = ["closed", "rag", "oracle", "router", "router_fallback"]
    jobs, jmap = [], []
    for it in items:
        for j, opt in enumerate(it["_opts"]):
            jobs.append((chain_prompt(it), " " + opt))
            jmap.append((it, "closed", j))

            art = (it["_text"] or "")[:a.ctx_chars]
            pre = (art + "\n\n") if art else ""
            jobs.append((pre + chain_prompt(it), " " + opt))
            jmap.append((it, "rag", j))

            jobs.append((hop2(it["bridge"], it), " " + opt))
            jmap.append((it, "oracle", j))

            if it["_picked"]:
                jobs.append((hop2(it["_picked"], it), " " + opt))
                jmap.append((it, "router", j))

    print(f"four conditions: {len(jobs):,} forward passes", file=sys.stderr)
    t0 = time.time()
    got = sc.score(jobs)
    print(f"  {time.time() - t0:.0f}s", file=sys.stderr)

    acc = defaultdict(lambda: defaultdict(dict))
    for (it, cond, j), s in zip(jmap, got):
        acc[id(it)][cond][j] = s
    for it in items:
        d = acc[id(it)]
        for cond in ("closed", "rag", "oracle", "router"):
            if cond not in d or len(d[cond]) < len(it["_opts"]):
                it[f"ans_{cond}"] = None
                it[f"hit_{cond}"] = 0
                continue
            pick = max(d[cond], key=lambda k: d[cond][k])
            it[f"ans_{cond}"] = it["_opts"][pick]
            it[f"hit_{cond}"] = int(pick == it["_gold"])
        # The router abstains on branch 3; the fallback guesses closed-book
        # there instead. Both are reported because they are different
        # products, not two readings of one number.
        if it["ans_router"] is None:
            it["ans_router_fallback"] = it["ans_closed"]
            it["hit_router_fallback"] = it["hit_closed"]
        else:
            it["ans_router_fallback"] = it["ans_router"]
            it["hit_router_fallback"] = it["hit_router"]

    chance = 100.0 / a.choices
    print(f"\n{'=' * 72}\nACCURACY   chance = {chance:.1f}%")
    acc_block("ALL ITEMS", items, conds)

    print("\n  by first-hop tier, which is where the argument lives")
    for lo, hi, lab in ((0, 2, "tier 0-2  bridge named"),
                        (3, 3, "tier 3    category named only"),
                        (4, 4, "tier 4    nothing named")):
        sel = [it for it in items if lo <= it["_tier"] <= hi]
        acc_block(lab, sel, conds)

    # The one comparison the paper turns on, tested as a paired comparison.
    t3 = [it for it in items if it["_tier"] == 3 and it["ans_router"] is not None]
    if t3:
        r_only = sum(1 for it in t3 if it["hit_router"] and not it["hit_rag"])
        g_only = sum(1 for it in t3 if it["hit_rag"] and not it["hit_router"])
        m = mcnemar(r_only, g_only)
        print(f"\n{'=' * 72}\nTIER 3: ROUTER AGAINST NAIVE RETRIEVAL   "
              f"n = {len(t3):,}")
        print(f"  router right, retrieval wrong   {r_only:,}")
        print(f"  retrieval right, router wrong   {g_only:,}")
        print(f"  discordant pairs {m['n_disc']:,}   chi2={m['chi']:.2f}   "
              f"p={m['p']:.3g}")
        if m["p"] >= 0.05:
            print("  not significant: on these items the ontology step does")
            print("  not beat putting the page in the prompt, and the paper")
            print("  has to say so.")

    print(f"\n{'=' * 72}\nBRIDGE IDENTIFICATION")
    for br in sorted(by_br):
        v = [it for it in by_br[br] if it["_picked"]]
        if not v:
            print(f"  {br:<10} no item reached a pick")
            continue
        p, lo, hi = wilson(sum(x["_bridge_right"] for x in v), len(v))
        ceil = 100 * sum(x["_cand_has_gold"] for x in v) / len(v)
        print(f"  {br:<10} {p:6.2f}% [{lo:5.2f}-{hi:5.2f}]  n={len(v):>5,}"
              f"   ceiling {ceil:6.2f}%")
    print("\n  The oracle and the router share the second-hop stem exactly,")
    print("  so the accuracy gap between them is this table and nothing else.")

    print(f"\n{'=' * 72}\nCOST")
    n = len(items)
    lk = sum(it["_lookups"] for it in items)
    print(f"  model forward passes per item   {sc.calls / n:8.1f}")
    print(f"  ontology lookups per item       {lk / n:8.1f}")
    print(f"  candidates scored per item      "
          f"{sum(len(it['_cands']) for it in items) / n:8.1f}")

    if a.out:
        cols = ["id", "a", "r1", "bridge", "r2", "answer", "tier",
                "branch", "picked", "bridge_right", "cand_n",
                "cand_has_gold", "lookups"]
        cols += [f"ans_{c}" for c in conds] + [f"hit_{c}" for c in conds]
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for it in items:
                rec = {k: it.get(k, "") for k in
                       ("id", "a", "r1", "bridge", "r2", "answer", "tier")}
                rec.update({"branch": it["_branch"], "picked": it["_picked"],
                            "bridge_right": it["_bridge_right"],
                            "cand_n": len(it["_cands"]),
                            "cand_has_gold": it["_cand_has_gold"],
                            "lookups": it["_lookups"]})
                for c in conds:
                    rec[f"ans_{c}"] = it.get(f"ans_{c}") or ""
                    rec[f"hit_{c}"] = it.get(f"hit_{c}", 0)
                w.writerow(rec)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
