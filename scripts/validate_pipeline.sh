#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "[CHECK] Python syntax"
python_files=()
while IFS= read -r -d '' file; do
  python_files+=("$file")
done < <(find src scripts/python -type f -name '*.py' -print0)
python -m py_compile "${python_files[@]}"

echo "[CHECK] Shell syntax"
while IFS= read -r -d '' file; do
  bash -n "$file"
done < <(find scripts -maxdepth 1 -type f -name '*.sh' -print0)

echo "[CHECK] No workstation-specific business paths"
if rg -n \
  '/home/linengfan/|Isaaclab/logs/tvae/|Isaaclab/23\.0URDF/|isaaclab_tasks\.manager_based\.locomotion\.velocity\.config\.sa_16dof' \
  . \
  --glob '!*.pt' \
  --glob '!*.STL' \
  --glob '!scripts/validate_pipeline.sh' \
  --glob '!checksums.sha256'; then
  echo "[FAIL] Found a workstation-specific path or legacy SA16 import." >&2
  exit 1
fi

echo "[CHECK] Required pipeline files"
required=(
  src/trot_bound_switch/tasks/expert_env_cfg.py
  src/trot_bound_switch/tasks/mdp/expert_bound_reward.py
  src/trot_bound_switch/tasks/mdp/expert_trot_reward.py
  src/trot_bound_switch/tasks/mdp/expert_curriculum.py
  src/trot_bound_switch/tasks/mdp/expert_symmetry.py
  scripts/python/train_expert.py
  scripts/python/collect_expert_data.py
  scripts/python/representation/train_shared_state_tvae.py
  scripts/python/representation/train_gait_action_decoders.py
  scripts/train_full_pipeline.sh
  scripts/play_pipeline.sh
  scripts/evaluate_pipeline.sh
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "[FAIL] Missing $file" >&2; exit 1; }
done

echo "[OK] Code-only pipeline validation passed. No simulator or training was started."
