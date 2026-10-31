# Joint current/future compact visual tokens

`visual_joint_aux` uses one set of compact queries to read two patch grids:
the current DINO feature and the World Model's predicted future DINO feature.
Each grid receives its own projection, the shared 2-D patch coordinates, and a
learned current/future frame embedding. The two contexts are concatenated over
the patch dimension, then passed through the existing two query blocks. The
output remains 8 x 128 tokens and uses the existing action-expert interaction
attention.

```text
current DINO --------------------┐
                                  ├─ projection + XY + frame id → concat → 8 queries
predicted future DINO -----------┘                                      ↓
current VLM context + state → query conditioning → compact tokens → action head
                                                                    ↘ current geometry loss
```

The auxiliary readout is supervised by the current-frame full geometry sidecar,
matching `visual_aux`. The future geometry sidecar is not loaded and no future
ground-truth image, state, or geometry enters the token encoder. The predicted
future feature is kept connected to the World Model, while the original direct
flow-conditioning path retains `detach_future_feature=true` in the matched
recipe. Inference only needs the current observation and the model prediction.

Generate and launch a matched Hanoi run:

```bash
conda activate lawam
cd /data/home/zyh/LaWAM
export LAWAM_ASSET_ROOT="$PWD"
export INTERACTION_DATA_ROOT=/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets
export INTERACTION_REVIEWED_ROOT="$PWD/results/interaction_sidecars_reviewed_20261025"
export INTERACTION_OUTPUT_ROOT="$PWD/results/Checkpoints/interaction_joint"
export INTERACTION_AUX_WEIGHT=0.1
unset FINGERTIP_ROOT
bash scripts/train_interaction_reviewed.sh hanoi visual_joint_aux --dry-run
bash scripts/train_interaction_reviewed.sh hanoi visual_joint_aux
```

For Microwave, replace `hanoi` with `microwave`. The command keeps the same
cohort, eight-GPU H30, 40k-step recipe as `visual_aux` and `visual_pred_aux`.
