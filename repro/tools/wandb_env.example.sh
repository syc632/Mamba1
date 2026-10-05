# Copy to wandb_env.sh and fill in your own key.
#   cp tools/wandb_env.sh.example tools/wandb_env.sh
# tools/run_125m.sh sources it if it exists; without it the run falls back to
# wandb offline mode (metrics are buffered locally and can be synced later).
#
# NEVER commit the real wandb_env.sh -- it is listed in .gitignore.
export WANDB_API_KEY=replace_me_with_your_own_key
export WANDB_PROJECT=${WANDB_PROJECT:-mamba1-repro}
export WANDB_SILENT=false
