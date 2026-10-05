"""Sample from a trained Mamba-125M checkpoint.

Loads a checkpoint written by train.py, runs a handful of fixed prompts through
the model, and prints the continuations. This is the qualitative counterpart to
the loss numbers: it answers "does the model actually produce coherent English".

    python sample.py --ckpt runs/125m/ckpt_final.pt
    python sample.py --ckpt runs/125m/ckpt_last.pt --greedy
"""
import argparse
import json
import os
import time

import torch
from tokenizers import Tokenizer

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

PRESETS = {
    "gpt3-125m": (768, 12), "gpt3-350m": (1024, 24),
    "gpt3-760m": (1536, 24), "gpt3-1.3b": (2048, 24),
}

PROMPTS = [
    "The capital of France is",
    "In physics, the law of conservation of energy states that",
    "The main difference between mitosis and meiosis is",
    "Water boils at 100 degrees Celsius because",
    "Once upon a time, there was a",
    "def quicksort(arr):",
    "The French Revolution began in 1789 and",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/125m/ckpt_final.pt")
    p.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    p.add_argument("--meta", default="data/fwedu.meta.json")
    p.add_argument("--model_size", default="gpt3-125m", choices=list(PRESETS))
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--max_new", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--greedy", action="store_true", help="deterministic argmax")
    p.add_argument("--device", default="cuda")
    p.add_argument("--prompt", action="append", default=None,
                   help="override the built-in prompt list (repeatable)")
    args = p.parse_args()

    d_model, n_layer = PRESETS[args.model_size]

    with open(args.meta) as f:
        meta = json.load(f)
    vocab_size = int(meta.get("vocab_size", 50257))
    eos = int(meta.get("eos", 50256))

    cfg = MambaConfig(
        d_model=d_model, n_layer=n_layer, vocab_size=vocab_size,
        ssm_cfg={"d_state": args.d_state, "d_conv": args.d_conv, "expand": args.expand},
        rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        pad_vocab_size_multiple=8, tie_embeddings=True,
    )
    model = MambaLMHeadModel(cfg, device=args.device, dtype=torch.float32).to(args.device)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    step = ck.get("step", "?")
    n_params = sum(q.numel() for q in model.parameters())
    print(f"[ckpt] {args.ckpt}  step={step}  params={n_params/1e6:.1f}M", flush=True)

    tok = Tokenizer.from_file(args.tokenizer)
    prompts = args.prompt or PROMPTS

    top_k = 1 if args.greedy else args.top_k
    top_p = 0.0 if args.greedy else args.top_p
    temp = 1.0 if args.greedy else args.temperature
    mode = "greedy" if args.greedy else f"temp={temp} top_p={top_p}"
    print(f"[mode] {mode}  max_new={args.max_new}\n", flush=True)

    for pr in prompts:
        ids = tok.encode(pr).ids
        x = torch.tensor([ids], device=args.device, dtype=torch.long)
        t0 = time.time()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = model.generate(
                    x, max_length=len(ids) + args.max_new,
                    top_k=top_k, top_p=top_p, temperature=temp,
                    vocab_size=vocab_size,  # never sample the padding rows
                    eos_token_id=eos, cg=False,
                )
        dt = time.time() - t0
        out = y[0].tolist()
        gen = out[len(ids):]
        tps = len(gen) / max(dt, 1e-9)
        text = tok.decode([t for t in out if t < vocab_size])
        print("=" * 78)
        print(f"PROMPT : {pr}")
        print(f"OUTPUT : {text}")
        print(f"[ {len(gen)} tok in {dt:.2f}s = {tps:.1f} tok/s ]", flush=True)
    print("=" * 78)


if __name__ == "__main__":
    main()
