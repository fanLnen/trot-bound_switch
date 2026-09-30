#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
cd "$REPO_ROOT"

CHECKPOINT="${1:-$REPO_ROOT/src/trot_bound_switch/pretrained/policy/model_1300.pt}"
if [[ $# -gt 0 ]]; then
  shift
fi

exec "${ISAACLAB_PYTHON[@]}" "$REPO_ROOT/scripts/python/evaluate.py" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$REPO_ROOT/outputs/evaluation" \
  --headless \
  --experience "$HEADLESS_EXPERIENCE" \
  "$@"
