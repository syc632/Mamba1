"""Mamba-1 reproduction trainer (arXiv:2312.00752) with wandb logging.

What is faithful to the paper (Table 12 + Appendix E.2.1):
  * AdamW, betas=(0.9, 0.95), weight decay 0.1, grad-clip 1.0, no dropout
  * linear LR warmup -> cosine decay to 1e-5, peak = 5x the GPT3 value
    ("improved recipe"). GPT3 values: 125M -> 6e-4, 350M -> 3e-4.
  * RMSNorm, no linear bias, tied embeddings, seq_len 2048
  * GPT2 tokenizer (vocab 50257)

What CANNOT be faithful on a single 8GB laptop GPU, and is therefore an
explicit CLI flag instead of a silent fudge:
  * batch size: paper uses 0.5M tokens/step; we use --global_batch_tokens
    (default 32768). That is the single biggest deviation.
  * total tokens: paper uses 2.5B (125M) / 7B (350M); we use --total_tokens.
  * corpus: paper uses the Pile; we use FineWeb-Edu (Pile is not reachable here).

Model size note -- two incompatible conventions exist in the paper:
  * "130m"/"370m"  = released checkpoints: 24 x 768 (129.1M) and 48 x 1024 (370.6M)
  * "gpt3-125m"    = Table 12 scaling-law spec: 12 x 768 (83.8M) and 24 x 1024 (211M)
Both are available via --model_size; the LR from Table 12 is attached to the
GPT3-spec name and reused for the checkpoint twin of the same label.

Usage:
  python repro/train.py --data data/fwedu --model_size 130m \
      --total_steps 20 --eval_every 10 --log_every 1
"""
import argparse
import json
import math
import os
import random
import signal
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# model presets
#   <name>: (d_model, n_layer, paper_lr)
#   lr comes from Table 12 (GPT3 spec peak LR), which is the value the
#   "improved recipe" then multiplies by 5.
# --------------------------------------------------------------------------
PRESETS = {
    # released-checkpoint geometry (what most people mean by "130M"/"370M")
    "130m": (768, 24, 6e-4),
    "370m": (1024, 48, 3e-4),
    "790m": (1536, 48, 2.5e-4),
    "1.4b": (2048, 48, 2e-4),
    # exact GPT3 / Table 12 scaling-law geometry
    "gpt3-125m": (768, 12, 6e-4),
    "gpt3-350m": (1024, 24, 3e-4),
    "gpt3-760m": (1536, 24, 2.5e-4),
    "gpt3-1.3b": (2048, 24, 2e-4),
}

PAPER_BATCH_TOKENS = 500_000
# Table 12 "Training steps" column
PAPER_STEPS = {
    "gpt3-125m": 4800, "gpt3-350m": 13500,
    "gpt3-760m": 29000, "gpt3-1.3b": 50000,
}


class TorchRMSNorm(nn.Module):
    """Pure-torch RMSNorm, used when the triton kernel is unavailable (Windows)."""

    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory = {"device": device, "dtype": dtype}
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory))
        self.eps = eps

    def forward(self, x):
        o = x.float()
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.eps)
        return (o * self.weight.float()).to(x.dtype)


def enable_grad_checkpointing(mixer):
    """v1.2.2 has no gradient-checkpointing flag; wrap each block manually."""
    from torch.utils.checkpoint import checkpoint

    def forward(self, input_ids, inference_params=None):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            if self.training:
                hidden_states, residual = checkpoint(
                    layer, hidden_states, residual,
                    inference_params=None, use_reentrant=False,
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            import mamba_ssm.models.mixer_seq_simple as mxs
            fn = mxs.rms_norm_fn if isinstance(self.norm_f, mxs.RMSNorm) \
                else mxs.layer_norm_fn
            hidden_states = fn(hidden_states, self.norm_f.weight, self.norm_f.bias,
                               eps=self.norm_f.eps, residual=residual,
                               prenorm=False, residual_in_fp32=self.residual_in_fp32)
        return hidden_states

    mixer.forward = forward.__get__(mixer, type(mixer))
    return mixer


def build_model(args, vocab_size):
    import mamba_ssm.models.mixer_seq_simple as mxs
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    triton_ok = mxs.RMSNorm is not None
    if not triton_ok:
        mxs.RMSNorm = TorchRMSNorm
        mxs.layer_norm_fn = None
        mxs.rms_norm_fn = None

    cfg = MambaConfig(
        d_model=args.d_model,
        n_layer=args.n_layer,
        vocab_size=vocab_size,
        ssm_cfg={"d_state": args.d_state, "d_conv": args.d_conv, "expand": args.expand},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=triton_ok,
        pad_vocab_size_multiple=8,
        tie_embeddings=True,
    )
    dtype = getattr(torch, args.param_dtype)
    model = MambaLMHeadModel(cfg, device=args.device, dtype=dtype)
    if args.grad_ckpt:
        enable_grad_checkpointing(model.backbone)
    return model, triton_ok, cfg


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
class TokenBin:
    """Memmap over the flat token files written by repro/data_prep.py.

    Prefers the explicit <prefix>_train.bin / <prefix>_val.bin pair; falls back
    to a single <prefix>.bin with an internal --val_frac holdout.
    """

    def __init__(self, prefix, seq_len, val_frac=0.01):
        self.seq_len = seq_len
        meta = {}
        meta_path = prefix + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        self.vocab_size = int(meta.get("vocab_size", 50257))
        dt = str(meta.get("dtype", "uint16"))
        self.dtype = np.dtype(np.uint16 if "uint16" in dt else np.int32)

        tr, va = prefix + "_train.bin", prefix + "_val.bin"
        if os.path.exists(tr) and os.path.exists(va):
            self.train = np.memmap(tr, dtype=self.dtype, mode="r")
            self.val = np.memmap(va, dtype=self.dtype, mode="r")
        else:
            allb = np.memmap(prefix + ".bin", dtype=self.dtype, mode="r")
            n_val = int(len(allb) * val_frac)
            self.train = allb[: len(allb) - n_val]
            self.val = allb[len(allb) - n_val:]
        self.n_train = max(0, len(self.train) - seq_len - 1)
        self.n_val = max(0, len(self.val) - seq_len - 1)
        print(f"[data] train={len(self.train)/1e6:.1f}M  val={len(self.val)/1e6:.2f}M  "
              f"vocab={self.vocab_size} seq_len={seq_len}", flush=True)
        if self.n_val <= 0:
            raise SystemExit("validation split is empty / shorter than seq_len")

    def _stack(self, arr, starts):
        sl = self.seq_len
        x = np.stack([np.asarray(arr[s:s + sl]) for s in starts])
        y = np.stack([np.asarray(arr[s + 1:s + sl + 1]) for s in starts])
        return (torch.from_numpy(x.astype(np.int64)),
                torch.from_numpy(y.astype(np.int64)))

    def train_batch(self, bsz, step=0, micro=0, seed=0):
        """Sample a training batch.

        The sample positions are derived from (seed, step, micro) instead of a
        module-level RNG. A global RNG gets reseeded on restart, which makes a
        resumed run silently re-consume the data it already saw at the beginning
        of training. Deriving per-step makes sampling stateless: step N always
        yields the same batch, before and after a resume.
        """
        rng = random.Random((seed * 1000003) ^ (step * 7919) ^ micro)
        return self._stack(self.train, [rng.randrange(self.n_train) for _ in range(bsz)])

    def val_batch(self, bsz, i=0):
        base = min(i * bsz * self.seq_len, max(0, self.n_val - bsz * self.seq_len))
        return self._stack(self.val, [base + k * self.seq_len for k in range(bsz)])


def make_lr_lambda(total_steps, warmup_steps, min_frac):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return min_frac + (1.0 - min_frac) * cos
    return fn


def compute_loss(logits, y, bf16_loss=False):
    if bf16_loss:
        # keeps the (B*L) x V tensors in bf16 instead of fp32: saves ~2/3 of
        # the LM-head activation memory, at the cost of fp32 softmax accuracy
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))


class _Stopper:
    """Cooperative stop: finish the current step, checkpoint, then exit.

    Killing the process mid-step discards every step since the last periodic
    save (up to --save_every steps of work). Handling the interrupt instead lets
    us persist first. A second signal aborts immediately without saving.
    """

    def __init__(self):
        self.requested = False

    def __call__(self, signum, frame):
        if self.requested:
            print("\n[stop] second signal - aborting now, nothing saved", flush=True)
            raise SystemExit(1)
        self.requested = True
        print("\n[stop] interrupt received - finishing current step, then "
              "checkpointing (hit again to abort immediately)", flush=True)


STOP = _Stopper()


def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data", required=True, help="prefix: <p>_train.bin / <p>_val.bin")
    p.add_argument("--val_frac", type=float, default=0.01)
    p.add_argument("--seq_len", type=int, default=2048)
    # model
    p.add_argument("--model_size", default="gpt3-125m", choices=list(PRESETS))
    p.add_argument("--paper", type=int, default=1,
                   help="1 = use Table 12 batch (0.5M tok) and step count verbatim")
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--param_dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--bf16_loss", action="store_true")
    # optimisation (paper defaults)
    p.add_argument("--lr", type=float, default=None, help="default: GPT3 value x5")
    p.add_argument("--improved_recipe", type=int, default=1)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--warmup_frac", type=float, default=0.01)
    # batching
    p.add_argument("--micro_bsz", type=int, default=1)
    p.add_argument("--global_batch_tokens", type=int, default=None,
                   help="default: 500000 with --paper, else 32768")
    p.add_argument("--total_tokens", type=float, default=None)
    p.add_argument("--total_steps", type=int, default=None,
                   help="default: Table 12 step count with --paper")
    p.add_argument("--bench_steps", type=int, default=0,
                   help="run this many steps and exit (throughput probe)")
    # bookkeeping
    p.add_argument("--out_dir", default=None)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=8)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--init_from", default=None)
    # wandb
    p.add_argument("--wandb_project", default="mamba1-repro")
    p.add_argument("--wandb_run", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_mode", default=None)
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    default_dm, default_nl, paper_lr = PRESETS[args.model_size]
    if args.d_model is None:
        args.d_model = default_dm
    if args.n_layer is None:
        args.n_layer = default_nl
    if args.lr is None:
        args.lr = paper_lr * (5.0 if args.improved_recipe else 1.0)
    if args.out_dir is None:
        args.out_dir = f"runs/{args.model_size}"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = TokenBin(args.data, args.seq_len, args.val_frac)
    model, triton_ok, cfg = build_model(args, ds.vocab_size)
    n_params = sum(q.numel() for q in model.parameters())

    decay, nodecay = [], []
    for _, prm in model.named_parameters():
        if prm.requires_grad:
            (nodecay if prm.ndim < 2 else decay).append(prm)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8, fused=True)

    micro_tokens = args.micro_bsz * args.seq_len
    if args.global_batch_tokens is None:
        args.global_batch_tokens = PAPER_BATCH_TOKENS if args.paper else 32768
    grad_accum = max(1, args.global_batch_tokens // micro_tokens)
    if args.total_steps is None:
        if args.paper and args.total_tokens is None and args.model_size in PAPER_STEPS:
            args.total_steps = PAPER_STEPS[args.model_size]
        else:
            args.total_steps = int((args.total_tokens or 7e8)
                                   // (grad_accum * micro_tokens))
    warmup = args.warmup_steps if args.warmup_steps is not None else \
        max(10, int(args.warmup_frac * args.total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_lambda(args.total_steps, warmup, args.min_lr / args.lr))

    step0 = 0
    if args.init_from and os.path.exists(args.init_from):
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step0 = int(ck.get("step", 0)) + 1
        for _ in range(step0):
            sched.step()
        print(f"[ckpt] resumed from {args.init_from} at step {step0}")

    if args.bench_steps:
        args.total_steps = step0 + args.bench_steps
        args.eval_every = 0
        args.save_every = 0
        args.log_every = 1

    print(f"[model] {args.model_size} d_model={args.d_model} n_layer={args.n_layer} "
          f"params={n_params/1e6:.1f}M dtype={args.param_dtype} "
          f"triton_rmsnorm={triton_ok} grad_ckpt={args.grad_ckpt}")
    plan_tokens = args.total_steps * grad_accum * micro_tokens
    print(f"[train] steps={args.total_steps} micro={args.micro_bsz}x{args.seq_len} "
          f"grad_accum={grad_accum} global_tokens/step={grad_accum*micro_tokens} "
          f"lr_peak={args.lr:.2e} warmup={warmup}")
    print(f"[plan] total_tokens={plan_tokens/1e9:.2f}B  "
          f"paper_batch={PAPER_BATCH_TOKENS} ratio={grad_accum*micro_tokens/PAPER_BATCH_TOKENS:.3f}  "
          f"paper_steps={PAPER_STEPS.get(args.model_size, '-')} "
          f"paper_tokens={PAPER_STEPS.get(args.model_size, 0)*PAPER_BATCH_TOKENS/1e9:.2f}B")
    # 500000/2048 = 244.14, so grad_accum=244 gives 499,712 tokens -- 0.06% off.
    # Only warn on real deviations, not on that rounding.
    batch_ok = abs(grad_accum * micro_tokens / PAPER_BATCH_TOKENS - 1.0) < 0.02
    steps_ok = abs(args.total_steps / PAPER_STEPS.get(args.model_size, args.total_steps)
                   - 1.0) < 0.02
    if not (batch_ok and steps_ok):
        print("[warn] deviating from Table 12 batch/steps (see --paper / "
              "--global_batch_tokens / --total_steps)")

    run = None
    if not args.no_wandb:
        import wandb
        mode = args.wandb_mode
        if mode is None:
            has_key = bool(os.getenv("WANDB_API_KEY")) or \
                os.path.exists(os.path.expanduser("~/.netrc"))
            mode = "online" if has_key else "offline"
            if mode == "offline":
                print("[wandb] no API key -> logging offline "
                      "(set WANDB_API_KEY, or `wandb sync` the offline run later)")
        cfg_dict = dict(vars(args))
        cfg_dict.update({
            "n_params": n_params,
            "grad_accum": grad_accum,
            "paper_batch_tokens": PAPER_BATCH_TOKENS,
            "batch_tokens_ratio_vs_paper": grad_accum * micro_tokens / PAPER_BATCH_TOKENS,
        })
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_run, mode=mode, config=cfg_dict)
        run.summary["params_M"] = n_params / 1e6

    amp_dtype = torch.bfloat16
    tokens_done = step0 * grad_accum * micro_tokens
    t_start = time.time()
    running = None
    torch.cuda.reset_peak_memory_stats()

    for sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
        try:
            signal.signal(getattr(signal, sig), STOP)
        except (AttributeError, ValueError, OSError):
            pass  # not available on this platform / not on the main thread

    completed = False
    step = step0 - 1  # keep `step` defined even if the loop body never runs
    for step in range(step0, args.total_steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        t0 = time.time()
        loss_acc = 0.0
        for mi in range(grad_accum):
            x, y = ds.train_batch(args.micro_bsz, step=step, micro=mi,
                                  seed=args.seed)
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x).logits
            loss = compute_loss(logits, y, args.bf16_loss)
            (loss / grad_accum).backward()
            loss_acc += loss.item() / grad_accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        sched.step()
        dt = time.time() - t0
        tokens_done += grad_accum * micro_tokens
        tps = grad_accum * micro_tokens / max(dt, 1e-9)
        running = loss_acc if running is None else 0.95 * running + 0.05 * loss_acc

        if run is not None:
            run.log({
                "train/loss": loss_acc,
                "train/loss_smooth": running,
                "train/lr": sched.get_last_lr()[0],
                "train/grad_norm": float(gn),
                "train/tokens_per_sec": tps,
                "train/tokens_seen": tokens_done,
                "sys/mem_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            }, step=step)

        if step % args.log_every == 0:
            eta = (args.total_steps - step) * dt / 3600
            print(f"step {step:>7}/{args.total_steps} loss {loss_acc:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} gn {float(gn):.2f} "
                  f"{tps:,.0f} tok/s mem {torch.cuda.max_memory_allocated()/1e9:.1f}G "
                  f"eta {eta:.1f}h", flush=True)

        if args.eval_every and (step % args.eval_every == 0
                                or step == args.total_steps - 1):
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for i in range(args.eval_iters):
                    x, y = ds.val_batch(args.micro_bsz, i)
                    x = x.to(args.device, non_blocking=True)
                    y = y.to(args.device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(x).logits
                    vl += compute_loss(logits, y, True).item()
            vl /= args.eval_iters
            print(f"[eval] step {step} val_loss {vl:.4f} ppl {math.exp(min(vl, 20)):.2f}",
                  flush=True)
            if run is not None:
                run.log({"val/loss": vl, "val/ppl": math.exp(min(vl, 20))}, step=step)

        if args.save_every and step and step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "args": vars(args)},
                       os.path.join(args.out_dir, "ckpt_last.pt"))
            print(f"[ckpt] saved step {step}", flush=True)

        if STOP.requested:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "args": vars(args)},
                       os.path.join(args.out_dir, "ckpt_last.pt"))
            print(f"[stop] checkpointed at step {step}, exiting cleanly", flush=True)
            break

    if step == args.total_steps - 1:
        completed = True

    if completed and not args.bench_steps:
        # Write to ckpt_final.pt, NOT ckpt_last.pt: overwriting the periodic
        # checkpoint would destroy the last resumable state. ckpt_last.pt keeps
        # the most recent periodic save so --init_from always has something valid.
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": args.total_steps - 1, "args": vars(args)},
                   os.path.join(args.out_dir, "ckpt_final.pt"))
        print(f"[ckpt] final saved (step {args.total_steps - 1}) "
              f"-> {os.path.join(args.out_dir, 'ckpt_final.pt')}", flush=True)
    print(f"[done] {time.time()-t_start:.0f}s tokens={tokens_done/1e9:.3f}B "
          f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}G", flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
