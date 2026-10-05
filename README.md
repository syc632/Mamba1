# Mamba-1 Reproduction

Reproduction of **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**
(Gu & Dao, arXiv:2312.00752) on a single consumer GPU, with a from-scratch training
script, wandb instrumentation, and a reported 125M-scale run.

This repo is pinned to the **Mamba-1** codebase (upstream tag `v1.2.2`, `mamba_ssm.__version__ == "1.2.2"`).
Everything added for Mamba-2 / Mamba-3 upstream — the SSD chunk-scan kernels, the
`mamba2` config classes, the `tilelang`/`quack-kernels` build stack, the ROCm patches,
the NGC CI workflows — has been removed. What is left is the original selective-scan
(S6) model plus the reproduction harness in `repro/`.

```
csrc/selective_scan/    17 files — the CUDA kernels (fwd/bwd for fp32/fp16/bf16, real & complex)
mamba_ssm/              14 files — Mamba / MambaLMHeadModel / MixerModel, Triton decode kernel
repro/                  this reproduction: data prep, training, sampling, reporting
```

---

## 1. What was reproduced

The paper's own scaling spec is **Table 12** (and Appendix E.2.1 for the optimizer
details). Following the paper rather than the released checkpoint names gives:

| | `gpt3-125m` (Table 12) | `gpt3-350m` (Table 12) |
|---|---|---|
| `n_layer` | 12 | 24 |
| `d_model` | 768 | 1024 |
| steps | 4 800 | 13 500 |
| tokens | 2.5 B | 7 B |
| GPT-3 peak LR | 6e-4 | 3e-4 |
| peak LR used (E.2.1 "improved recipe" = ×5) | **3e-3** | 1.5e-3 |

Common: `d_state=16`, `d_conv=4`, `expand=2`, 0.5 M tokens/step, `seq_len=2048`,
AdamW `(0.9, 0.95)`, `wd=0.1`, grad-clip 1.0, no dropout, linear warmup (1 % of steps)
then cosine decay to 1e-5.

> **Naming trap.** The released checkpoints `130m` / `370m` are **24×768** and
> **48×1024** — twice the layer count of the Table 12 spec. `--model_size` accepts
> both conventions; this reproduction used the Table 12 one.

**Corpus.** The paper trains on the Pile. The Pile is not reachable from this
machine, so the run used **FineWeb-Edu** (tokenised with the GPT-2 BPE tokenizer,
same 50257 vocab) — a like-for-like swap of corpus, not of recipe.

---

## 2. Environment

Verified on Windows 11 / **RTX 5060 Laptop, 8 GB** (sm_120 / Blackwell).

| | |
|---|---|
| Python | 3.11.15 (conda env `mamba_formal`) |
| PyTorch | 2.12.1+cu132 |
| CUDA | 13.2 (nvcc 13.2.78) |
| Triton | 3.7.1 |
| transformers | 5.14.1 |

```bash
conda create -n mamba_formal python=3.11 -y
conda activate mamba_formal

# torch from the Tsinghua mirror (CUDA build)
pip install torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install triton transformers datasets einops wandb -i https://pypi.tuna.tsinghua.edu.cn/simple

# build the selective-scan extension from source (no prebuilt wheel exists for sm_120)
set MAMBA_FORCE_BUILD=TRUE
pip install . --no-build-isolation
```

`setup.py` builds for `compute_120` only: CUDA 13 dropped `sm_53/62/70/72`, so the
upstream hard-coded arch list fails outright on Blackwell.

The hard dependency on the compiled kernel (`selective_scan_cuda`) cannot be avoided —
`mamba_ssm/ops/selective_scan_interface.py` imports it unconditionally. A pure-PyTorch
fallback exists in that file but is roughly an order of magnitude slower and materialises
the full `(B, D, N, L)` scan, which does not fit in 8 GB at `L=2048`.

---

## 3. Data

```bash
python repro/data_prep.py --out data/fwedu
```

Streams FineWeb-Edu, tokenises with GPT-2 BPE, and writes `<prefix>_train.bin` /
`<prefix>_val.bin` (memmapped `uint32`; 24 GB total here, ~1 % held out for validation).

---

## 4. Training

```bash
cp repro/tools/wandb_env.example.sh repro/tools/wandb_env.sh   # then paste your key
bash repro/tools/run_125m.sh
```

which is

```bash
python repro/train.py \
  --data data/fwedu --model_size gpt3-125m --paper 1 \
  --total_steps 4800 --micro_bsz 1 --seq_len 2048 \
  --d_state 16 --d_conv 4 --expand 2 \
  --lr 3e-3 --min_lr 1e-5 --wd 0.1 --beta1 0.9 --beta2 0.95 --clip 1.0 \
  --warmup_frac 0.01 \
  --log_every 10 --eval_every 200 --eval_iters 8 --save_every 400 \
  --out_dir runs/125m --wandb_run mamba125m-paper
```

0.5 M tokens/step at `micro_bsz=1, seq_len=2048` means 244 gradient-accumulation
micro-steps per optimizer step.

**Resuming** is supported and tested: `--init_from runs/125m/ckpt_last.pt` restores
model + optimizer + step. `train.py` installs `SIGINT`/`SIGTERM`/`SIGBREAK` handlers
that finish the current step and checkpoint before exiting, so `Ctrl-C` is safe.
On Windows the launcher must send `CTRL_BREAK_EVENT` — `CTRL_C_EVENT` is ignored by
processes started with `CREATE_NEW_PROCESS_GROUP`.

**wandb** records `loss`, `ema_loss`, `val_loss`, `val_ppl`, `lr`, `grad_norm`,
`tokens`, `throughput_tok_s`, `sys/mem_peak_gb`, `sys/tokens_total`.

---

## 5. Result (125M, completed run)

| metric | value |
|---|---|
| wall clock | 84 471 s (23.5 h) |
| tokens | 2.399 B |
| final train loss | 3.14 |
| final val loss / PPL | **3.285 / 26.71** |
| avg throughput | 22 946 tok/s |
| peak VRAM | 3.69 GB |
| MFU | ≈ 44 % of measured bf16 peak |

Validation perplexity: 38 058 (step 0) → 97.7 (200) → 42.6 (800) → 37.7 (1200) →
30.87 (2400) → **26.71** (final). The early part of that curve is invisible on a linear
axis; the report plots it log-scaled.

**Parameter count is 83.9 M, not 125 M.** `lm_head` is tied to the embedding
(`lm_head.weight.data_ptr() == embedding.weight.data_ptr()`), so the 38.6 M embedding
is counted once. 83.86 M + 38.60 M (untied head) + 1.57 M (GPT-3 positional embeddings,
which Mamba does not have) ≈ 124 M — which is how the paper arrives at "125M".

**Peak memory breakdown:** 1.342 GB static (params+grads+Adam m,v = 16 B/param),
0.206 GB logits in bf16, 0.412 GB cross-entropy in fp32, ~1.72 GB activations/workspace.

---

## 6. Sampling and report

```bash
python repro/sample.py --ckpt runs/125m/ckpt_final.pt --max_new 80
python repro/make_report.py --runs <wandb_run_id> --gen runs/gen_final.txt \
                            -o Mamba125M_report.html
bash repro/tools/finish.sh        # both of the above
```

`Mamba125M_report.html` (Chart.js inlined under `repro/assets/`, so it opens offline)
is committed as the run artifact.

**Generation quality.** Register and syntax are fully learned — output reads like
FineWeb-Edu textbook prose. Factual recall is unreliable ("the capital of France is
Poitiers"), and code generation mostly fails (`def quicksort(arr):` drifts off task
after two lines). Expected at 2.4 B tokens on a 84 M model.

Note that **training and inference use different kernels**: training calls the CUDA
`selective_scan_cuda`, while `generate()` calls the Triton `selective_state_update`.
A model that trains fine can still break at generation time if the Triton path is
broken.

---

## 7. Bugs found and fixed during the reproduction

Compile time (sm_120 / CUDA 13 / MSVC):

1. CCCL removed `cub::` APIs the kernels use → substituted equivalents.
2. MSVC `C2975` on template arguments in the scan kernels.
3. CUDA 13 no longer supports `sm_53/62/70/72` → `setup.py` builds `compute_120` only.
4. Missing headers under the CUDA 13 layout.

Runtime:

5. **Final checkpoint overwrote `ckpt_last.pt`**, destroying the last resumable state →
   final save now goes to `ckpt_final.pt`.
6. **Resumed runs silently re-consumed early data.** Batch sampling used a module-level
   RNG that got reseeded by `random.seed(args.seed)` on restart. Positions are now
   derived from `(seed, step, micro)`, making sampling stateless. Verified by
   `tools/ckpt_roundtrip_test.sh` (< 1e-4 loss difference).
7. `triton.language.math.log1p` was removed in Triton 3.7.1 → `tl.log(1.0 + tl.exp(dt))`
   in `mamba_ssm/ops/triton/selective_state_update.py`.
8. transformers 5.x degrades the HF output classes to bare `object`, and `decode()`
   calls `output_cls(sequences=..., scores=...)` → replaced with a real dataclass in
   `mamba_ssm/utils/generation.py`.

---

## 8. Not reproduced

- The **350M** run (13 500 steps, 7 B tokens) — ~4× the 125M cost, not run.
- Downstream zero-shot evaluation on the lm-evaluation-harness suites.
- The Pile corpus (unreachable); FineWeb-Edu substituted.

## 9. References

- Paper: https://arxiv.org/abs/2312.00752
- Upstream: https://github.com/state-spaces/mamba (this repo tracks tag `v1.2.2`)
- License: Apache 2.0 (`LICENSE`) — the reproduction harness in `repro/` is under the same license.
