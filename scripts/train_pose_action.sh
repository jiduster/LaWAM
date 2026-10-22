#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/tmp/lawam_pose_hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lawam_pose_mpl}"
mode="${1:-orange}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$mode" != orange && "$mode" != baseline ]]; then
  echo 'Usage: bash scripts/train_pose_action.sh [baseline|orange] [training overrides]' >&2
  exit 2
fi
steps="${POSE_STEPS:-10000}"
seed="${POSE_SEED:-2026}"
config="results/pose_action_configs/${mode}_${steps}_seed${seed}_$(date +%Y%m%d_%H%M%S).yaml"
python scripts/configure_pose_action.py --mode "$mode" --output "$config" \
  --steps "$steps" --seed "$seed" \
  --sidecar-root "${POSE_SIDECAR_ROOT:-$PWD/results/object_pose_sidecars}" \
  --dataset-root "${POSE_DATA_ROOT:-/data/home/zyh/dexjoco/outputs/photograph_object_pose_20261022/lerobot/bimanual_photograph}"
bash train_lawam.sh --config_yaml "$config" "$@"
