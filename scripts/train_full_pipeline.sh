#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/artifacts/full_pipeline}"

cat <<EOF
[PIPELINE] Outputs: $ARTIFACT_ROOT
[PIPELINE] 1/5 Bound expert
EOF
"$SCRIPT_DIR/train_bound_expert.sh"

echo "[PIPELINE] 2/5 Trot expert"
"$SCRIPT_DIR/train_trot_expert.sh"

echo "[PIPELINE] 3/5 Expert datasets"
"$SCRIPT_DIR/collect_expert_datasets.sh"

echo "[PIPELINE] 4/5 Shared TVAE and gait decoders"
"$SCRIPT_DIR/train_representation.sh"

echo "[PIPELINE] 5/5 Forced transition PPO"
"$SCRIPT_DIR/train_transition_from_pipeline.sh"

echo "[PIPELINE] Complete: $ARTIFACT_ROOT"
