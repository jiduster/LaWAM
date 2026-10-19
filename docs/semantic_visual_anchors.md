# Current-image semantic visual anchors

This first version needs only the existing RGB observations, language, proprioception,
and expert actions. It does not require object poses, depth, masks, fingertip sidecars,
new checkpoints, or per-task keypoint annotations. It reuses the policy's frozen DINO
features; it does not download or replace the existing vision encoder.

## Computation

1. Read current primary-image DINO patch tokens only (`h_t_original`).
2. Normalize descriptors and concatenate normalized patch-center UV coordinates.
3. Deterministically initialize 16 cluster centers by farthest-point selection, then
   perform four k-means updates on descriptor/UV features.
4. Pool descriptors in each region and select an actual patch closest to its center.
5. Encode the descriptor and UV along with masked pooled VLM context and masked
   proprioceptive state. All anchors share the same encoder, without task IDs or
   anchor-index embeddings. A learned importance distribution produces a weighted
   task/state-dependent context for the anchor tokens.
6. Feed these tokens through an independent residual cross-attention branch after
   action-expert blocks 4, 8, 12, 16 (one-based). All flow integration updates use
   the same current-observation tokens. Existing visual and WM conditions remain.

The action-attention output projection is initialized to zero. Loading a compatible
baseline preserves its original actions at initialization for the same random noise.
BC/flow loss trains the new encoders and attention; this version adds no auxiliary
loss. During the first update, the zero output projection blocks upstream gradients
through this residual; after it moves away from zero BC reaches the encoders.

## Scope

These are **2D image-region candidates**, not measured 3D object contact points.
Candidates cover the whole primary image, including possible hand/background regions.
The first version does not require SAM or assume object masks are already available.
Clustering is annotation-free; action learning still uses expert demonstrations.
No ground-truth future image enters proposals or tokens; future images remain used
by the existing WM/distillation losses. No correspondence across time is assumed.
The module conditions on normalized proprioception, not explicit metric TCP-to-anchor
offsets, because this version has no depth/camera calibration.

## Training

In the existing `lawam` environment and a job with the desired GPUs:

```bash
cd /data/home/zyh/LaWAM
NUM_PROCESSES=8 TRAIN_STEPS=10000 bash scripts/train_visual_anchors.sh
```

This starts from the saved C_h30 Stage 2 policy (30 actions, 44 action dimensions),
loads both its action expert and LAM, retains saved normalization statistics, and
restarts the optimizer/scheduler for 10k additional steps. New modules use 1e-4,
existing flow/LAM decoder use 1e-5, and VLM uses 2e-6. Logging uses local train.log;
W&B is disabled by default to avoid network/service startup dependencies.

Matched continuation control (same starting checkpoint, steps, rates, data and seed):

```bash
NUM_PROCESSES=8 TRAIN_STEPS=10000 ANCHOR_MODE=control bash scripts/train_visual_anchors.sh
```

Outputs are separated under `.../dexjoco_bimanual_photograph_C_h30_visual_anchors/`
and `.../dexjoco_bimanual_photograph_C_h30_visual_control/`. Checkpoints are saved
every 1k steps, with `final_model/pytorch_model.pt` at completion.

Use `INITIAL_CHECKPOINT=/path/to/pytorch_model.pt` to override the initializer.
This is weights-only continuation, not optimizer-state resumption. Extra arguments
after the script name are forwarded to `train_lawam.sh`, including multi-task data
and trainer overrides. `--dry-run` prints the launch command without starting GPUs.
Do not compare directly with older controls that used different continuation schedules.

The launch script preserves job-assigned CUDA visibility. On the Pair Slurm cluster,
run it inside the assigned GPU job; physical GPU indices and CUDA ordinals can differ.

## Inference and inspection

The normal `serve_lawam` command works with a new checkpoint: its saved run config
constructs the anchor encoder and attention before checkpoint weights are loaded.
Use the matching updated LaWAM code on the inference machine. No new observation
fields or server CLI flags are required. Do not delete the parent run `config.yaml`
or `dataset_statistics.json` when moving the final-model file.

`backend.predict_action(..., return_intermediates=True)` additionally returns:

- `visual_anchor_uv`: `[B,K,2]`, primary-image patch centers in [-1,1].
- `visual_anchor_patch_indices`: `[B,K]`, row-major DINO patch indices.
- `visual_anchor_importance`: `[B,K]`, learned normalized weights.

These weights describe the learned selector, not calibrated contact probabilities
or the action expert's actual attention weights.

Preview current DINO proposals on existing images or a dataset video before training:

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  python scripts/preview_visual_anchors.py --video /path/to/episode.mp4 \
  --frames 0 120 240 --output-dir /tmp/lawam_anchor_preview
```

This defaults to CPU and loads only the frozen DINO encoder, not the VLM/action expert.
It restores vision weights from the Stage 2 initializer and uses the same current-image
feature layer and candidate algorithm as training. PNG overlays and `anchors.json`
contain candidate locations; they do not show learned relevance or contact labels.
For standalone RGB images, replace `--video ... --frames ...` with `--images a.png b.png`.

## Verification

```bash
PYTHONPATH=. NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=1 \
  python -m unittest scripts.test_semantic_visual_anchors
```

The tests exercise deterministic spatial proposals, zero-init equivalence, BC gradient
flow, changes in actions when anchor tokens change, both DiT variants, CFG dropping,
training/inference parity, no future leakage, diagnostics, optimizer grouping, and
saved config/checkpoint reload. They use tiny CPU models, not a full robot rollout.
