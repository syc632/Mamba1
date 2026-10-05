#!/usr/bin/env bash
# Paper-exact Mamba-125M run (Table 12 + Appendix E.2.1).
#
#   n_layer=12, d_model=768, seq_len=2048
#   4800 steps x 0.5M tokens = 2.4B tokens
#   peak LR 3e-3  (= GPT3 6e-4 x 5, improved recipe)
#   AdamW b=(0.9,0.95), wd=0.1, clip=1.0, no dropout
#   linear warmup 48 steps -> cosine decay to 1e-5
#
# Measured on RTX 5060 Laptop: ~19.9k tok/s, 3.7 GB, ~33.5 h for 2.4B tokens.
set -eu
cd "$(dirname "$0")/.."

HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/d/anaconda3/envs/mamba_formal/python.exe
DATA=${DATA:-data/fwedu}
STEPS=${STEPS:-4800}
RUN_NAME=${RUN_NAME:-mamba125m-paper}

# wandb credentials (WANDB_API_KEY); without this the run falls back to offline
[ -f "$HERE/wandb_env.sh" ] && . "$HERE/wandb_env.sh"
export WANDB_PROJECT=${WANDB_PROJECT:-mamba1-repro}

exec "$PY" train.py \
  --data "$DATA" \
  --model_size gpt3-125m \
  --paper 1 \
  --total_steps "$STEPS" \
  --micro_bsz 1 \
  --seq_len 2048 \
  --d_state 16 --d_conv 4 --expand 2 \
  --lr 3e-3 --min_lr 1e-5 \
  --wd 0.1 --beta1 0.9 --beta2 0.95 --clip 1.0 \
  --warmup_frac 0.01 \
  --log_every 10 --eval_every 200 --eval_iters 8 --save_every 400 \
  --out_dir runs/125m \
  --wandb_run "$RUN_NAME" \
  "$@"
