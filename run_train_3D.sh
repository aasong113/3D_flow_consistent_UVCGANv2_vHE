#!/usr/bin/env bash
set -euo pipefail

# Runs: train_3D_embedding_style_loss_TM.py with the args from your CLI snippet.
#
# Notes:
# - W&B: Prefer exporting WANDB_API_KEY (or running `wandb login`) instead of hardcoding it.
# - `--no-embedding-loss` is a backwards-compatible alias that disables `--use-style-fusion`.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"
mkdir -p "$WANDB_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_PATH="${SCRIPT_PATH:-/home/durrlab/Desktop/Anthony/UGVSM/3D_flow_consistent_UVCGANv2_vHE/scripts/20251213_Inverted_combined_BIT2HE_normal_duodenum_crypts/train_3D_embedding_style_loss_TM.py}"
ROOT_DATA_PATH="${ROOT_DATA_PATH:-/home/durrlab/Desktop/Anthony/data/20251225_duodenum_crypts}"

exec "$PYTHON_BIN" "$SCRIPT_PATH" \
  --root_data_path "$ROOT_DATA_PATH" \
  --batch-size 1 \
  --z-spacing 2 \
  --lambda-sub-loss 0 \
  --lambda-embedding-loss 1.0 \
  --no-embedding-loss \
  --lambda-style-fusion 0.0 \
  --style-fusion-inject adain \
  --wandb \
  --wandb-entity sanhong113 \
  --wandb-mode online \
  "$@"

