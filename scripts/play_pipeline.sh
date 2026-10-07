#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
cd "$REPO_ROOT"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/full_pipeline}"
CHECKPOINT="${1:-$ARTIFACT_ROOT/transition/model_1300.pt}"
if [[ $# -gt 0 ]]; then
  shift
fi

exec "${ISAACLAB_PYTHON[@]}" "$REPO_ROOT/scripts/python/play.py" \
  --checkpoint "$CHECKPOINT" \
  --trot_decoder "$ARTIFACT_ROOT/representation/action_decoders/trot/best.pt" \
  --bound_decoder "$ARTIFACT_ROOT/representation/action_decoders/bound/best.pt" \
  --experience "$GUI_EXPERIENCE" \
  "$@"
