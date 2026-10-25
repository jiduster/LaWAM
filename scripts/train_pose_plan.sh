#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HOME="${HF_HOME:-/tmp/lawam_pose_hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lawam_pose_mpl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
: "${POSE_DATA_ROOT:?Set POSE_DATA_ROOT to the audited bimanual_photograph dataset directory}"
: "${FINGERTIP_ROOT:?Set FINGERTIP_ROOT to the directory containing bimanual_photograph/manifest.json}"
steps="${POSE_STEPS:-40000}"
seed="${POSE_SEED:-2026}"
config="results/pose_plan_configs/green_${steps}_seed${seed}_$(date +%Y%m%d_%H%M%S)_$$.yaml"
python scripts/configure_pose_plan.py \
  --data-root "$POSE_DATA_ROOT" --fingertip-root "$FINGERTIP_ROOT" \
  --asset-root "${LAWAM_ASSET_ROOT:-$PWD}" \
  --output-root "${POSE_OUTPUT_ROOT:-$PWD/results/pose_stage2_green_local}" \
  --output "$config" --steps "$steps" --seed "$seed"
if [[ "${1:-}" == --dry-run ]]; then
  exit 0
fi
bash train_lawam.sh --config_yaml "$config" "$@"
