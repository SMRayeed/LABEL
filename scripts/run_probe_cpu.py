#!/usr/bin/env python3
"""
CPU probe for proposal 1's go/no-go. No GPU, no generation, no SLURM.

Instead of asking the model to write an answer and checking the string, this
scores the log-probability the model assigns to the true target against a
handful of distractors drawn from the same relation, and records whether the
true one wins. One forward pass per option, nothing generated.

That is cheaper and it is also the better instrument here. Generation makes
the measurement depend on formatting and on instruction-following, which a
small model does badly and which has nothing to do with the hypothesis.
Forced choice isolates what the question is actually about: whether the fact
is in the weights. It is what the knowledge-probing literature uses, and it
measures encoding rather than retrieval, which is the distinction Calderon et
al. show matters.

    python3 run_probe_cpu.py --edges p1_edges.csv --out p1_scored.csv \\
        --model Qwen/Qwen2.5-0.5B-Instruct

    python3 gonogo_p1.py regress --scored p1_scored.csv --outcome mc_correct

1,800 edges at 5 options each is 9,000 short forward passes. On a login node
with a 0.5B model that is a few minutes.

Model size is a knob, not a constant: --model Qwen/Qwen2.5-1.5B-Instruct costs
a few times more and is still CPU-feasible. Running two sizes also gives the
scale comparison the proposal wants, for free.

Needs transformers and torch. CPU only by default.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

# The stem names the source and the relation and stops. The options are the
# continuations being scored, so the target must never appear in the stem.
STEMS = {
    "indication": "{s} is a drug used to treat the disease called",
    "contraindication": "{s} is a drug that must not be given to patients who have",
    "off-label use": "{s} is a drug sometimes used off-label to treat",
    # Hetionet's two drug-disease relations. "treats" addresses the disease
    # itself; "palliates" only relieves its symptoms, so the stems have to
    # differ or the probe cannot tell the two apart.
    "treats": "{s} is a drug used to treat the disease called",
    "palliates": "{s} is a drug used to relieve the symptoms of",
    "eats": "The species {s} feeds on the species",
    "preysOn": "The species {s} preys on the species",
    "parasiteOf": "The species {s} is a parasite of the species",
    "pollinates": "The species {s} pollinates the plant",
    "visitsFlowersOf": "The species {s} visits the flowers of",
    "hasHost": "The species {s} has as its host the species",
}
GENERIC_STEM = "{s} {r}"


def stem_for(rel):
    return STEMS.get(rel, GENERIC_STEM)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges", required=True, help="p1_edges.csv from the build stage")
    ap.add_argument("--out", default="p1_scored.csv")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--choices", type=int, default=5,
                    help="options per item, one true and the rest distractors")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads; 0 leaves the default")
    ap.add_argument("--score", choices=["mean", "sum"], default="mean",
                    help="mean log-prob per token (default) does not favor "
                         "short answers; sum does")
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N edges")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", action="store_true", help="use CUDA if present")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                    help="weights dtype; float32 reproduces the paper, bfloat16 "
                         "halves memory for 14B-class models on a GPU")
    a = ap.parse_args()

    if not os.path.exists(a.edges):
        sys.exit(f"{a.edges} is missing. Run gonogo_p1.py build first.")

    try:
        import torch
    except BaseException as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"could not import torch: {msg}\n", file=sys.stderr)
        if "libtorch_cuda" in str(e) or "nccl" in str(e).lower() or "undefined symbol" in str(e):
            print(
                "That is a broken CUDA build of torch, not a missing GPU: the\n"
                "CUDA library cannot resolve a symbol against the NCCL package\n"
                "installed beside it. It fails on a compute node too.\n\n"
                "This script only needs CPU torch. The clean fix is a separate\n"
                "environment, which leaves your existing one untouched:\n\n"
                "  python -m venv ~/cpu_probe_env\n"
                "  source ~/cpu_probe_env/bin/activate\n"
                "  pip install --index-url https://download.pytorch.org/whl/cpu torch\n"
                "  pip install transformers\n\n"
                "Then run this script from that environment. On OSC a module\n"
                "may also work: module spider pytorch, then load one and skip\n"
                "the pip install.", file=sys.stderr)
        sys.exit(1)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except BaseException as e:
        sys.exit(f"could not import transformers: {type(e).__name__}: {e}")

    if a.threads:
        torch.set_num_threads(a.threads)
    device = "cuda" if (a.gpu and torch.cuda.is_available()) else "cpu"
    print(f"device: {device}, torch threads: {torch.get_num_threads()}",
          file=sys.stderr)

    rows = list(csv.DictReader(open(a.edges, encoding="utf-8")))
    if a.limit:
        rows = rows[:a.limit]
    rng = random.Random(a.seed)

    # Distractors come from the same relation, so the choice is between
    # plausible entities of the right type rather than between a disease and a
    # city. Targets this source is actually paired with elsewhere in the file
    # are excluded; the sample cannot see partners outside it, so a small
    # residual chance remains that a distractor is a true partner. With
    # roughly 600 targets per relation and a handful of partners per source
    # that is on the order of one percent, and it costs accuracy equally
    # across tiers, so it cannot create a tier effect.
    pool = defaultdict(list)
    partners = defaultdict(set)
    for r in rows:
        pool[r["relation"]].append(r["target"])
        partners[(r["relation"], r["source"])].add(r["target"])
    pool = {k: sorted(set(v)) for k, v in pool.items()}

    items = []
    for r in rows:
        rel, src, tgt = r["relation"], r["source"], r["target"]
        cand = pool[rel]
        if len(cand) < a.choices + 2:
            continue
        bad = partners[(rel, src)]
        opts = [tgt]
        guard = 0
        while len(opts) < a.choices and guard < 200:
            guard += 1
            c = rng.choice(cand)
            if c not in bad and c not in opts:
                opts.append(c)
        if len(opts) < a.choices:
            continue
        order = list(range(len(opts)))
        rng.shuffle(order)
        items.append({"row": r, "options": [opts[i] for i in order],
                      "gold": order.index(0)})

    if not items:
        sys.exit("no scorable items; is the edge file empty?")
    print(f"{len(items):,} items x {a.choices} options = "
          f"{len(items) * a.choices:,} forward passes", file=sys.stderr)

    print(f"loading {a.model} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # transformers 5.x renamed torch_dtype to dtype; accept either.
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model,
                                                     dtype=getattr(torch, a.dtype))
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model,
                                                     torch_dtype=getattr(torch, a.dtype))
    model = model.to(device)
    model.eval()

    # Flatten to (item index, option index, prompt, continuation).
    flat = []
    for i, it in enumerate(items):
        stem = stem_for(it["row"]["relation"]).format(
            s=it["row"]["source"], r=it["row"]["relation"])
        for j, opt in enumerate(it["options"]):
            flat.append((i, j, stem, " " + opt))

    scores = [[0.0] * a.choices for _ in items]
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(flat), a.batch_size):
            chunk = flat[b:b + a.batch_size]
            seqs, spans = [], []
            for _i, _j, stem, cont in chunk:
                p_ids = tok(stem, add_special_tokens=True)["input_ids"]
                c_ids = tok(cont, add_special_tokens=False)["input_ids"]
                seqs.append(p_ids + c_ids)
                spans.append((len(p_ids), len(c_ids)))
            width = max(len(s) for s in seqs)
            pad = tok.pad_token_id
            input_ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            attn = torch.zeros((len(seqs), width), dtype=torch.long)
            for k, s in enumerate(seqs):                 # right padding
                input_ids[k, :len(s)] = torch.tensor(s, dtype=torch.long)
                attn[k, :len(s)] = 1
            input_ids, attn = input_ids.to(device), attn.to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            for k, (i, j, _s, _c) in enumerate(chunk):
                n_p, n_c = spans[k]
                if n_c == 0:
                    scores[i][j] = -1e30
                    continue
                # the token at position t is predicted by the logits at t-1
                tot = 0.0
                for t in range(n_p, n_p + n_c):
                    tot += float(logprobs[k, t - 1, input_ids[k, t]])
                scores[i][j] = tot / n_c if a.score == "mean" else tot
            done = min(b + a.batch_size, len(flat))
            rate = done / max(1e-9, time.time() - t0)
            left = (len(flat) - done) / max(1e-9, rate)
            sys.stderr.write(f"  {done:,}/{len(flat):,}  {rate:.0f}/s  "
                             f"~{left / 60:.1f} min left    \r")
            sys.stderr.flush()
    sys.stderr.write(" " * 58 + "\r")
    sys.stderr.flush()

    out_rows = []
    for it, sc in zip(items, scores):
        pick = max(range(len(sc)), key=lambda j: sc[j])
        r = dict(it["row"])
        r["mc_correct"] = int(pick == it["gold"])
        r["mc_gold_logprob"] = f"{sc[it['gold']]:.4f}"
        r["mc_best_logprob"] = f"{sc[pick]:.4f}"
        r["mc_margin"] = f"{sc[it['gold']] - max(s for j, s in enumerate(sc) if j != it['gold']):.4f}"
        r["mc_n_choices"] = len(sc)
        out_rows.append(r)

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    acc = sum(r["mc_correct"] for r in out_rows) / len(out_rows)
    chance = 1.0 / a.choices
    print(f"\nwrote {a.out}: {len(out_rows):,} items")
    print(f"  accuracy {100 * acc:.2f}%   chance {100 * chance:.2f}%")
    by = defaultdict(list)
    for r in out_rows:
        by[int(r["tier"])].append(r["mc_correct"])
    lab = {0: "exact name", 1: "+synonym", 2: "+common name",
           3: "ancestor only", 4: "not named at all"}
    print("\n  accuracy by tier")
    for k in range(5):
        v = by.get(k, [])
        if v:
            print(f"    tier {k} {lab[k]:<18} {100 * sum(v) / len(v):>6.2f}%  "
                  f"n={len(v):>5,}")
    if acc < chance * 1.2:
        print("\n  Accuracy is close to chance. A model this small may not hold")
        print("  these facts at all, which leaves no variation for the")
        print("  regression. Try a larger --model before reading the verdict.")
    print(f"\nNext:\n  python3 gonogo_p1.py regress --scored {a.out} "
          f"--outcome mc_correct")


if __name__ == "__main__":
    main()
