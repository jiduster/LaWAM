#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
CONFIG="starVLA/config/training/starvla_train_dexjoco_bimanual_photograph_C_h30_visual_anchors.yaml"
MODE="${ANCHOR_MODE:-anchors}"
STEPS="${TRAIN_STEPS:-10000}"
CHECKPOINT="${INITIAL_CHECKPOINT:-results/Checkpoints/dexjoco_bimanual_photograph_C_h30/20261015_001205+dexjoco_bimanual_photograph_C_h30_sft/final_model/pytorch_model.pt}"
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi
case "${MODE}" in
  anchors) ENABLED=true ;;
  control) ENABLED=false ;;
  *) echo "ANCHOR_MODE must be anchors or control" >&2; exit 2 ;;
esac
if [[ ! "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_STEPS must be a positive integer" >&2
  exit 2
fi
if [[ -d "${CHECKPOINT}" ]]; then
  CHECKPOINT="${CHECKPOINT%/}/pytorch_model.pt"
fi
for required in "${CONFIG}" "${CHECKPOINT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file not found: ${required}" >&2
    exit 1
  fi
done

# Preserve CUDA_VISIBLE_DEVICES assigned by the user's current GPU job.
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export HF_HOME="${HF_HOME:-/tmp/lawam_hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_lawam}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

ARGS=(
  --config_yaml "${CONFIG}"
  --run_root_dir "results/Checkpoints/dexjoco_bimanual_photograph_C_h30_visual_${MODE}"
  --run_id "C_h30_visual_${MODE}_continue_${STEPS}"
  --framework.action_model.visual_anchor_conditioning "${ENABLED}"
  --trainer.pretrained_checkpoint "${CHECKPOINT}"
  --trainer.max_train_steps "${STEPS}"
)
if [[ "${MODE}" == control ]]; then
  ARGS+=(--trainer.learning_rate.visual_anchors.modules '[]')
fi
echo "Visual anchor experiment: mode=${MODE}; additional steps=${STEPS}; processes=${NUM_PROCESSES}"
echo "Initialize weights from: ${CHECKPOINT}; optimizer and scheduler restart"
if [[ "${DRY_RUN}" == true ]]; then
  printf '%q ' bash train_lawam.sh "${ARGS[@]}" "$@"
  printf '\n'
else
  exec bash train_lawam.sh "${ARGS[@]}" "$@"
fi
