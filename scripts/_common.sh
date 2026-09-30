#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEADLESS_EXPERIENCE="$REPO_ROOT/apps/trot_bound_switch.python.headless.kit"
GUI_EXPERIENCE="$REPO_ROOT/apps/trot_bound_switch.python.kit"

if [[ -n "${ISAACLAB_ROOT:-}" ]]; then
  ISAACLAB_ROOT="$(cd "$ISAACLAB_ROOT" && pwd)"
else
  ISAACLAB_ROOT="$(cd "$REPO_ROOT/../Isaaclab" 2>/dev/null && pwd || true)"
fi

if [[ ! -x "$ISAACLAB_ROOT/isaaclab.sh" ]]; then
  echo "IsaacLab was not found. Set ISAACLAB_ROOT to the existing IsaacLab checkout." >&2
  exit 1
fi
if [[ ! -f "$HEADLESS_EXPERIENCE" || ! -f "$GUI_EXPERIENCE" ]]; then
  echo "Repository-local Kit experience files are missing." >&2
  exit 1
fi

has_isaaclab_python() {
  "$@" -c 'import sys, importlib.metadata as m; assert (3, 11) <= sys.version_info[:2] < (3, 12); m.version("isaaclab"); m.version("isaacsim")' \
    >/dev/null 2>&1
}

CONDA_ENV_PYTHON=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$CONDA_BASE" ]]; then
    CONDA_ENV_PYTHON="$CONDA_BASE/envs/env_isaaclab/bin/python"
  fi
fi

if [[ -n "${TROT_BOUND_PYTHON:-}" ]]; then
  ISAACLAB_PYTHON=("$TROT_BOUND_PYTHON")
elif has_isaaclab_python python; then
  ISAACLAB_PYTHON=(python)
elif [[ -x "$CONDA_ENV_PYTHON" ]] && has_isaaclab_python "$CONDA_ENV_PYTHON"; then
  ISAACLAB_PYTHON=("$CONDA_ENV_PYTHON")
elif command -v conda >/dev/null 2>&1 && \
     has_isaaclab_python conda run --no-capture-output -n env_isaaclab python; then
  ISAACLAB_PYTHON=(conda run --no-capture-output -n env_isaaclab python)
else
  cat >&2 <<'EOF'
No compatible IsaacLab Python environment was found.
Activate the existing IsaacLab environment, or set TROT_BOUND_PYTHON to its Python executable.
EOF
  exit 1
fi

export PYTHONUNBUFFERED=1
export REPO_ROOT ISAACLAB_ROOT HEADLESS_EXPERIENCE GUI_EXPERIENCE
