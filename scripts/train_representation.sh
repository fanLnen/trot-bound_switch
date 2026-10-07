#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
cd "$REPO_ROOT"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/full_pipeline}"
REP="$REPO_ROOT/scripts/python/representation"
DATA="$ARTIFACT_ROOT/datasets"
INDEX="$ARTIFACT_ROOT/representation/index"
SPLIT="$INDEX/split_norm"
TVAE="$ARTIFACT_ROOT/representation/state_tvae"
DECODERS="$ARTIFACT_ROOT/representation/action_decoders"

"${ISAACLAB_PYTHON[@]}" "$REP/audit_expert_dataset.py" --dataset "$DATA/trot"
"${ISAACLAB_PYTHON[@]}" "$REP/audit_expert_dataset.py" --dataset "$DATA/bound"

"${ISAACLAB_PYTHON[@]}" "$REP/build_sequence_index.py" \
  --trot_dataset "$DATA/trot" \
  --bound_dataset "$DATA/bound" \
  --output_dir "$INDEX"

"${ISAACLAB_PYTHON[@]}" "$REP/prepare_split_normalization.py" \
  --index_dir "$INDEX" \
  --output_dir "$SPLIT" \
  --val_fraction 0.10 \
  --seed 42

"${ISAACLAB_PYTHON[@]}" "$REP/train_shared_state_tvae.py" \
  --index_dir "$INDEX" \
  --split_norm_dir "$SPLIT" \
  --output_dir "$TVAE" \
  --epochs 30 \
  --batch_size 500 \
  --lr 0.001 \
  --beta_kl 0.1 \
  --latent_dim 16 \
  --d_model 64 \
  --nhead 8 \
  --num_layers 2 \
  --dim_feedforward 64 \
  --seed 42 \
  --device cuda:0 \
  --checkpoint_every 10 \
  --latent_snapshot_per_gait 10000

"${ISAACLAB_PYTHON[@]}" "$REP/train_gait_action_decoders.py" \
  --encoder_checkpoint "$TVAE/best.pt" \
  --index_dir "$INDEX" \
  --split_norm_dir "$SPLIT" \
  --output_dir "$DECODERS" \
  --gait both \
  --epochs 20 \
  --batch_size 500 \
  --lr 0.001 \
  --hidden_dim 64 \
  --checkpoint_every 5 \
  --seed 42 \
  --device cuda:0
