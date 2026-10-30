#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HOME="${HF_HOME:-$PWD/results/.hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
if [[ "$NUM_PROCESSES" != 8 ]]; then
  echo 'Matched recipe requires NUM_PROCESSES=8 (4 samples/rank, accumulation=1, global batch=32).' >&2
  exit 1
fi
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/train_interaction.sh {hanoi|microwave} {baseline|oracle|visual|visual_aux|visual_pred_aux|visual_wm_aux|visual_wm_aux_detach} [--dry-run]' >&2
  exit 1
fi
task="$1"
variant="$2"
shift 2
: "${INTERACTION_DATA_ROOT:?Set INTERACTION_DATA_ROOT to the LeRobot dataset parent}"
config_root="${INTERACTION_CONFIG_ROOT:-$PWD/results/interaction_configs}/$(date +%Y%m%d_%H%M%S)_$$"
args=(--task "$task" --variant "$variant" --data-root "$INTERACTION_DATA_ROOT"
      --asset-root "${LAWAM_ASSET_ROOT:-$PWD}" --config-root "$config_root"
      --output-root "${INTERACTION_OUTPUT_ROOT:-$PWD/results/Checkpoints/interaction}"
      --steps "${INTERACTION_STEPS:-40000}" --seed "${INTERACTION_SEED:-2026}")
args+=(--aux-weight "${INTERACTION_AUX_WEIGHT:-0.1}")
[[ -z "${INTERACTION_SIDECAR_ROOT:-}" ]] || args+=(--sidecar-root "$INTERACTION_SIDECAR_ROOT")
[[ -z "${INTERACTION_EPISODES:-}" ]] || args+=(--episodes "$INTERACTION_EPISODES")
[[ -z "${FINGERTIP_ROOT:-}" ]] || args+=(--fingertip-root "$FINGERTIP_ROOT")
[[ -z "${LAWAM_PRETRAIN:-}" ]] || args+=(--pretrain "$LAWAM_PRETRAIN")
[[ -z "${LAWAM_LAM:-}" ]] || args+=(--lam "$LAWAM_LAM")
[[ -z "${LAWAM_VLM:-}" ]] || args+=(--vlm "$LAWAM_VLM")
[[ -z "${LAWAM_DINO:-}" ]] || args+=(--dino "$LAWAM_DINO")
[[ -z "${LAWAM_LAM_YAML:-}" ]] || args+=(--lam-yaml "$LAWAM_LAM_YAML")
python scripts/configure_interaction.py "${args[@]}"
case "$task" in
  hanoi) task=bimanual_hanoi ;;
  microwave) task=bimanual_microwave_cook ;;
esac
if [[ "${1:-}" == --dry-run ]]; then exit 0; fi
bash train_lawam.sh --config_yaml "$config_root/$task/$variant.yaml" "$@"
