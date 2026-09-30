#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
cd "$REPO_ROOT"

exec "${ISAACLAB_PYTHON[@]}" "$REPO_ROOT/scripts/python/train.py" \
  --task SA-16DOF-FuXian-Trot-Bound-Transition-v0 \
  --num_envs 4096 \
  --max_iterations 1351 \
  --run_name forced_transition_formal_4096x1351_reproduction \
  --headless \
  --experience "$HEADLESS_EXPERIENCE" \
  "$@"
