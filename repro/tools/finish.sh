#!/usr/bin/env bash
# Run once the training loop has written runs/125m/ckpt_final.pt.
#   1. sample generations from the final checkpoint
#   2. build the HTML report from the merged wandb history
set -eu
cd "$(dirname "$0")/.."
PY=/d/anaconda3/envs/mamba_formal/python.exe

if [ ! -f runs/125m/ckpt_final.pt ]; then
  echo "runs/125m/ckpt_final.pt not found - training still running?" >&2
  exit 1
fi

echo "=== sampling ==="
"$PY" sample.py --ckpt runs/125m/ckpt_final.pt --max_new 80 2>&1 \
  | grep -vE "custom_f|custom_b|Warning|warn" | tee runs/gen_final.txt

echo "=== report ==="
"$PY" make_report.py --runs pr9w23uf lntugrhw \
  --gen runs/gen_final.txt -o Mamba125M_report.html 2>&1 \
  | grep -vE "wandb:|warn|futurew"

echo "=== done ==="
ls -la Mamba125M_report.html
