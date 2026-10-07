#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
cd "$REPO_ROOT"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/full_pipeline}"
TROT_DECODER="$ARTIFACT_ROOT/representation/action_decoders/trot/best.pt"
BOUND_DECODER="$ARTIFACT_ROOT/representation/action_decoders/bound/best.pt"

for decoder in "$TROT_DECODER" "$BOUND_DECODER"; do
  if [[ ! -f "$decoder" ]]; then
    echo "Missing trained decoder: $decoder" >&2
    exit 1
  fi
done

exec "${ISAACLAB_PYTHON[@]}" "$REPO_ROOT/scripts/python/train.py" \
  --task SA-16DOF-FuXian-Trot-Bound-Transition-v0 \
  --num_envs 4096 \
  --max_iterations 1351 \
  --seed 42 \
  --run_name full_pipeline_transition_from_scratch \
  --trot_decoder "$TROT_DECODER" \
  --bound_decoder "$BOUND_DECODER" \
  --output_dir "$ARTIFACT_ROOT/transition" \
  --headless \
  --experience "$HEADLESS_EXPERIENCE" \
  "$@"
