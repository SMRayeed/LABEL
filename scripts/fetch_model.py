#!/usr/bin/env python3
"""
Download a model into the Hugging Face cache, on the login node.

Compute nodes have no internet, so the download has to happen here first and
the batch job then reads it offline. This writes to one explicit cache
directory and prints the path, so the batch script and this script cannot
disagree about where the weights are.

    python3 fetch_model.py
    python3 fetch_model.py --model Qwen/Qwen2.5-3B-Instruct
    python3 fetch_model.py --cache $HOME/hf_cache

Re-running is cheap: anything already present is skipped.
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--cache", default="",
                    help="HF_HOME to use (default: $HOME/hf_cache)")
    ap.add_argument("--check", action="store_true", default=True,
                    help="load the tokenizer afterwards to prove it is usable")
    ap.add_argument("--no-check", dest="check", action="store_false")
    a = ap.parse_args()

    cache = a.cache or os.path.join(os.path.expanduser("~"), "hf_cache")
    os.environ["HF_HOME"] = cache
    os.makedirs(cache, exist_ok=True)

    # A vision-language checkpoint has a different text stack, so its scores
    # do not sit on the same scale as a plain text model of another size.
    # Mixing the two silently would ruin the across-size comparison, so say
    # so here rather than let it surface as a strange number later.
    if "-VL-" in a.model or a.model.lower().endswith("-vl"):
        print(f"WARNING: {a.model} looks like a vision-language checkpoint.\n"
              f"         For text log-probability scoring use the plain text\n"
              f"         model, e.g. Qwen/Qwen2.5-7B-Instruct.\n",
              file=sys.stderr)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is missing: pip install huggingface_hub")

    print(f"model  {a.model}")
    print(f"cache  {cache}")
    print("downloading (already-present files are skipped) ...")

    try:
        path = snapshot_download(a.model)
    except Exception as e:
        sys.exit(f"\ndownload failed: {type(e).__name__}: {e}\n"
                 f"If this is a 401 or 403, the repo is gated and needs\n"
                 f"  huggingface-cli login\n"
                 f"If it is a network error, this must run on a LOGIN node;\n"
                 f"compute nodes have no outbound internet.")

    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    print(f"\nweights at {path}")
    print(f"size       {total / 1e9:.1f} GB")

    if a.check:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.model)
            n = len(tok("the quick brown fox")["input_ids"])
            print(f"tokenizer  loads, {n} tokens on a test string")
        except Exception as e:
            print(f"tokenizer  FAILED to load: {type(e).__name__}: {e}",
                  file=sys.stderr)
            sys.exit(1)

    print(f"\nNext:\n  MODEL={a.model} sbatch run_probe.sh")


if __name__ == "__main__":
    main()
