#!/usr/bin/env python3
"""Save a tiny random causal language model with a chat template.

Used by `bash run_all.sh --smoke` so the model steps run end to end offline
in seconds. Its answers are random; only the plumbing is under test.

    python3 tiny_model.py --text smoke/work/wiki_cache --out smoke/tiny_model_a
"""

import argparse
import glob
import os

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
import torch

CHAT = ("{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="a folder of .txt files to train the tokenizer on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    files = sorted(glob.glob(os.path.join(a.text, "*.txt")))[:200]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=800, special_tokens=["<s>", "</s>", "<pad>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train(files, tr)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>",
                                   eos_token="</s>", pad_token="<pad>")
    fast.chat_template = CHAT
    cfg = LlamaConfig(vocab_size=fast.vocab_size, hidden_size=32,
                      intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2,
                      max_position_embeddings=32768,
                      bos_token_id=fast.bos_token_id, eos_token_id=fast.eos_token_id,
                      pad_token_id=fast.pad_token_id)
    model = LlamaForCausalLM(cfg)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    fast.save_pretrained(a.out)
    print(f"tiny model at {a.out}: vocab {fast.vocab_size}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")


if __name__ == "__main__":
    main()
