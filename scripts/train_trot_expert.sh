#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
cd "$REPO_ROOT"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/full_pipeline}"
exec "${ISAACLAB_PYTHON[@]}" "$REPO_ROOT/scripts/python/train_expert.py" \
  --gait trot \
  --num_envs 4096 \
  --max_iterations 2101 \
  --seed 42 \
  --run_name trot_expert_from_scratch_final_turning_recipe \
  --output_dir "$ARTIFACT_ROOT/experts/trot" \
  --headless \
  --experience "$HEADLESS_EXPERIENCE" \
  "$@"
