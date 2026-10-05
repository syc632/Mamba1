#!/usr/bin/env bash
# Verify checkpoint save/resume correctness.
#   run A : scratch, 7 steps  -> reference trajectory (loss + lr at every step)
#   run B : scratch, 6 steps  -> ckpt saved at step 5
#   run C : resume from B     -> step0=6, runs step 6 only
# C's step-6 lr must EXACTLY equal A's step-6 lr; loss must be close (CUDA
# kernels are not bit-deterministic, so allow a small tolerance).
# Tiny model so it can run alongside the real 125M training.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
PY=/d/anaconda3/envs/mamba_formal/python.exe
TMP=runs/_ckpttest
mkdir -p "$TMP"

COMMON="--data data/fwedu --model_size gpt3-125m --d_model 64 --n_layer 2 \
  --micro_bsz 1 --seq_len 512 --global_batch_tokens 2048 \
  --warmup_steps 2 --lr 3e-3 --min_lr 1e-5 \
  --log_every 1 --eval_every 0 --no_wandb --seed 0 --device cuda"

echo "=================== RUN A (scratch, 7 steps) ==================="
"$PY" train.py $COMMON --total_steps 7 --save_every 0 --out_dir "$TMP/a" 2>&1 \
  | grep -E "^step" | tee "$TMP/a.log"

# B must use the SAME --total_steps as A, otherwise the cosine schedule differs
# and the two trajectories are not comparable. The final state now goes to
# ckpt_final.pt, so ckpt_last.pt stays at step 5 and remains resumable.
echo "=================== RUN B (scratch, 7 steps, save every 5) ==================="
"$PY" train.py $COMMON --total_steps 7 --save_every 5 --out_dir "$TMP/b" 2>&1 \
  | grep -E "^step|ckpt" | tee "$TMP/b.log"

echo "=================== RUN C (resume from B, runs step 6) ==================="
"$PY" train.py $COMMON --total_steps 7 --save_every 0 \
  --init_from "$TMP/b/ckpt_last.pt" --out_dir "$TMP/c" 2>&1 \
  | grep -E "^step|resumed" | tee "$TMP/c.log"

echo "=================== CHECK ==================="
"$PY" - <<'PYEOF'
import re, torch
def parse(p):
    out = {}
    for line in open(p):
        m = re.match(r"step\s+(\d+)/\d+ loss ([\d.]+) lr ([\d.eE+-]+)", line)
        if m:
            out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out
A, B, C = parse("runs/_ckpttest/a.log"), parse("runs/_ckpttest/b.log"), parse("runs/_ckpttest/c.log")

ok = True
print("run A steps:", sorted(A), " run B steps:", sorted(B), " run C steps:", sorted(C))

if 6 not in C:
    print("FAIL: run C did not execute step 6 (resume produced no work)")
    ok = False
else:
    la, lra = A[6]; lc, lrc = C[6]
    print(f"\nstep 6 lr    A={lra:.6e}   C={lrc:.6e}   -> {'OK' if lra==lrc else 'MISMATCH'}")
    if lra != lrc: ok = False
    print(f"step 6 loss  A={la:.4f}       C={lc:.4f}       -> diff {abs(la-lc):.5f}")
    if abs(la - lc) > 0.02:
        print("FAIL: resumed loss diverges from scratch trajectory")
        ok = False
    else:
        print("OK: resumed loss continues the trajectory (small delta = kernel nondeterminism)")

# A vs B: same seed, same config -> measures pure run-to-run nondeterminism
d = max(abs(A[s][0]-B[s][0]) for s in sorted(set(A) & set(B)))
print(f"\nrun-to-run scratch nondeterminism (A vs B, max loss delta): {d:.5f}")

ck = torch.load("runs/_ckpttest/b/ckpt_last.pt", map_location="cpu", weights_only=False)
print("ckpt_last keys:", sorted(ck.keys()), " saved step:", ck["step"])
if ck["step"] != 5:
    print(f"FAIL: ckpt_last.pt holds step {ck['step']}, expected 5 (final save must not clobber it)")
    ok = False
fin = "runs/_ckpttest/b/ckpt_final.pt"
import os
if not os.path.exists(fin):
    print("FAIL: ckpt_final.pt missing")
    ok = False
else:
    cf = torch.load(fin, map_location="cpu", weights_only=False)
    print(f"ckpt_final step: {cf['step']} (expected 6)")
    if cf["step"] != 6: ok = False
nstate = len(ck["opt"].get("state", {}))
has_momentum = nstate > 0 and "exp_avg" in next(iter(ck["opt"]["state"].values()))
print(f"optimizer state tensors: {nstate}, has exp_avg: {has_momentum}")
if not has_momentum:
    print("FAIL: AdamW momenta not persisted -> resume would lose optimizer state")
    ok = False

print("\nRESULT:", "PASS" if ok else "FAIL")
PYEOF
