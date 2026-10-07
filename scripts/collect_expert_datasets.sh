#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
cd "$REPO_ROOT"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/full_pipeline}"
BOUND_CHECKPOINT="${BOUND_EXPERT_CHECKPOINT:-$ARTIFACT_ROOT/experts/bound/model_1350.pt}"
TROT_CHECKPOINT="${TROT_EXPERT_CHECKPOINT:-$ARTIFACT_ROOT/experts/trot/model_2100.pt}"

for checkpoint in "$BOUND_CHECKPOINT" "$TROT_CHECKPOINT"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing expert checkpoint: $checkpoint" >&2
    exit 1
  fi
done

"${ISAACLAB_PYTHON[@]}" "$REPO_ROOT/scripts/python/collect_expert_data.py" \
  --gait bound \
  --checkpoint "$BOUND_CHECKPOINT" \
  --output_dir "$ARTIFACT_ROOT/datasets/bound" \
  --num_envs 512 \
  --target_samples 2000000 \
  --chunk_size 100000 \
  --seed 42 \
  --headless \
  --experience "$HEADLESS_EXPERIENCE"

"${ISAACLAB_PYTHON[@]}" "$REPO_ROOT/scripts/python/collect_expert_data.py" \
  --gait trot \
  --checkpoint "$TROT_CHECKPOINT" \
  --output_dir "$ARTIFACT_ROOT/datasets/trot" \
  --num_envs 512 \
  --target_samples 2000000 \
  --chunk_size 100000 \
  --seed 42 \
  --headless \
  --experience "$HEADLESS_EXPERIENCE"
