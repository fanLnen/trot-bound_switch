#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
cd "$REPO_ROOT"

exec "${ISAACLAB_PYTHON[@]}" "$REPO_ROOT/scripts/python/play.py" \
  --checkpoint "$REPO_ROOT/src/trot_bound_switch/pretrained/policy/model_1300.pt" \
  --experience "$GUI_EXPERIENCE" \
  "$@"
