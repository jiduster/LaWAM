#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: bash scripts/train_geometry_auxonly_10k.sh [--dry-run]" >&2
  exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="starVLA/config/training/starvla_train_dexjoco_bimanual_photograph_C_h30_geometry.yaml"
INITIAL_CHECKPOINT="results/Checkpoints/dexjoco_bimanual_photograph_C_h30/20261015_001205+dexjoco_bimanual_photograph_C_h30_sft/final_model/pytorch_model.pt"
for required_file in "${CONFIG}" "${INITIAL_CHECKPOINT}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

RUN_ROOT="results/Checkpoints/dexjoco_bimanual_photograph_C_h30_geometry_auxonly"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID_3K="C_h30_geometry_auxonly_continue_3k"
RUN_ID_7K="C_h30_geometry_auxonly_continue_to_10k"
OUTPUT_3K="${RUN_ROOT}/${RUN_TIMESTAMP}+${RUN_ID_3K}"
OUTPUT_7K="${RUN_ROOT}/${RUN_TIMESTAMP}+${RUN_ID_7K}"
CHECKPOINT_3K="${OUTPUT_3K}/final_model/pytorch_model.pt"
for output_dir in "${OUTPUT_3K}" "${OUTPUT_7K}"; do
  if [[ -e "${output_dir}" ]]; then
    echo "Output already exists: ${output_dir}" >&2
    exit 1
  fi
done

# Preserve the GPU visibility assigned by the current job.
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export HF_HOME="${HF_HOME:-/tmp/lawam_hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_lawam}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

COMMON_ARGS=(
  --run_root_dir "${RUN_ROOT}"
  --is_debug false
  --seed 2026
  --framework.action_model.geometry_conditioning true
  --framework.action_model.geometry_inject false
  --framework.action_model.geometry_loss_weight 0.05
  --framework.action_model.fingertip_loss_weight 0.0
  --trainer.learning_rate.action_model.lr 1e-4
  --trainer.learning_rate.action_model.modules '[policy_backend.geometry_head]'
  --trainer.learning_rate.existing_flow.modules '[policy_backend.flow]'
  --trainer.load_pretrained_policy_flow true
  --trainer.load_pretrained_policy_lam true
  --trainer.use_pretrained_dataset_statistics true
  --trainer.num_warmup_steps 100
  --trainer.save_interval 1000
  --trainer.eval_interval 1000
  --trainer.eval_batches 5
  --trainer.learning_rate.world_model.lr 1e-5
  --trainer.learning_rate.vlm.lr 2e-6
)

run_phase() {
  local phase_name="$1"
  shift
  echo "${phase_name}"
  if [[ "${DRY_RUN}" == true ]]; then
    printf 'LAWAM_RUN_TIMESTAMP=%q ' "${RUN_TIMESTAMP}"
    printf '%q ' bash train_lawam.sh "$@"
    printf '\n'
  else
    LAWAM_RUN_TIMESTAMP="${RUN_TIMESTAMP}" bash train_lawam.sh "$@"
  fi
}

run_phase "Phase 1: 3000 additional steps with geometry loss only" \
  --config_yaml "${CONFIG}" "${COMMON_ARGS[@]}" \
  --run_id "${RUN_ID_3K}" \
  --trainer.pretrained_checkpoint "${INITIAL_CHECKPOINT}" \
  --trainer.max_train_steps 3000 \
  --trainer.learning_rate.base 1e-5 \
  --trainer.learning_rate.existing_flow.lr 1e-5

if [[ "${DRY_RUN}" == false && ! -f "${CHECKPOINT_3K}" ]]; then
  echo "Phase 1 did not produce its final checkpoint: ${CHECKPOINT_3K}" >&2
  exit 1
fi

run_phase "Phase 2: 7000 more steps with geometry loss only" \
  --config_yaml "${CONFIG}" "${COMMON_ARGS[@]}" \
  --run_id "${RUN_ID_7K}" \
  --trainer.pretrained_checkpoint "${CHECKPOINT_3K}" \
  --trainer.max_train_steps 7000 \
  --trainer.learning_rate.base 5e-5 \
  --trainer.learning_rate.existing_flow.lr 2e-5

echo "Final checkpoint: ${OUTPUT_7K}/final_model/pytorch_model.pt"
