#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

"${ISAACLAB_PYTHON[@]}" -c 'import sys; print(f"[INFO] Using Python: {sys.executable} ({sys.version.split()[0]})")'
"${ISAACLAB_PYTHON[@]}" -m pip install --no-deps --no-build-isolation -e "$REPO_ROOT"
echo "[OK] Installed trot-bound-switch into the existing IsaacLab Python environment."
echo "[INFO] IsaacLab root: $ISAACLAB_ROOT"
echo "[INFO] Repository root: $REPO_ROOT"
