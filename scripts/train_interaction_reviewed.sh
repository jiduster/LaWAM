#!/usr/bin/env bash
# Matched experiments using explicitly accepted reconstructed sidecars.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/train_interaction_reviewed.sh {hanoi|microwave} {baseline|visual|visual_aux|visual_pred_aux|visual_joint_aux|oracle|visual_wm_aux|visual_wm_aux_detach} [--dry-run]' >&2
  exit 1
fi
case "$1" in
  hanoi|bimanual_hanoi) dataset=bimanual_hanoi ;;
  microwave|bimanual_microwave_cook) dataset=bimanual_microwave_cook ;;
  *) echo "Unsupported task: $1" >&2; exit 1 ;;
esac
export INTERACTION_DATA_ROOT="${INTERACTION_DATA_ROOT:-/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets}"
export INTERACTION_OUTPUT_ROOT="${INTERACTION_OUTPUT_ROOT:-$PWD/results/Checkpoints/interaction_reviewed}"
export INTERACTION_SIDECAR_ROOT="${INTERACTION_REVIEWED_ROOT:-$PWD/results/interaction_sidecars_reviewed_20261025}"
export INTERACTION_EPISODES="$INTERACTION_SIDECAR_ROOT/$dataset/accepted_episodes.json"
if [[ ! -f "$INTERACTION_EPISODES" ]]; then
  echo "Missing reviewed cohort: $INTERACTION_EPISODES; run scripts/prepare_interaction_candidates.py first." >&2
  exit 1
fi
bash scripts/train_interaction.sh "$@"
