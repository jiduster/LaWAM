#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export HF_HOME="${HF_HOME:-/tmp/lawam_hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_lawam}"
export NO_ALBUMENTATIONS_UPDATE=1

mode="${1:-oracle}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$mode" != oracle && "$mode" != control ]]; then
  echo "Usage: bash scripts/train_oracle_contact.sh [oracle|control] [training overrides]" >&2
  exit 2
fi
sidecars="${CONTACT_SIDECAR_ROOT:-/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_contact_sidecars}"
dataset_root="${CONTACT_DATA_ROOT:-/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets}"
enabled=true
if [[ "$mode" == oracle ]]; then
  python scripts/prepare_photograph_contact.py \
    --dataset-root "$dataset_root/bimanual_photograph" \
    --output-root "$sidecars" --validate-only
else
  enabled=false
fi
bash train_lawam.sh \
  --config_yaml starVLA/config/training/starvla_train_dexjoco_bimanual_photograph_C_h30_contact.yaml \
  --run_id "C_h30_contact_${mode}_continue_10k" \
  --run_root_dir "results/Checkpoints/dexjoco_bimanual_photograph_C_h30_contact_${mode}" \
  --framework.action_model.flow_cfg.contact_conditioning "$enabled" \
  --framework.action_model.contact_sidecar_root "$sidecars" \
  --datasets.vla_data.data_root_dir "$dataset_root" \
  "$@"
