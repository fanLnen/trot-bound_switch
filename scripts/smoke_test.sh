#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
cd "$REPO_ROOT"

exec "${ISAACLAB_PYTHON[@]}" "$REPO_ROOT/scripts/python/smoke_test.py" \
  --headless \
  --experience "$HEADLESS_EXPERIENCE" \
  "$@"
