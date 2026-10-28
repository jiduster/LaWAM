# Future object geometry supervision (2026-10-28)

`visual_wm_aux` reads the **predicted** future DINO patches with per-entity
queries and supervises future object state. The geometry loss backpropagates
through the readout, LaWM decoder and predicted latent-action branch.
`visual_wm_aux_detach` has identical parameters and forward values at fixed
weights, but detaches only the readout input. Its geometry loss updates only
the readout; original Stage 2 losses still train the policy.

These are standalone additions to the original LaWM baseline. Neither includes
the previous current-frame visual interaction-token branch, and neither adds
tokens or predicted poses to the action expert. The head is not evaluated at
inference, and no pose payload or sidecar is needed for inference. Use the
saved checkpoint config so that model architecture and keys match.

The matched recipe keeps `detach_future_feature=true` on the existing flow
conditioning branch in **both** variants. That setting is independent of the
new auxiliary-only detach. The attached geometry loss uses `shared.h_t1_pred`
directly, before flow conditioning detaches it.

## Labels and targets

Reuse `results/interaction_sidecars_reviewed_20261025`, with its existing
reconstructed-label acceptance policy and original provenance. Hanoi uses
98 episodes (97 train / 1 validation); Microwave uses 100 (99 / 1). The cohort
and assets match the previous baseline/visual/visual_aux/oracle recipe.

Labels use the last **actually sampled WM video frame**. With two images at
video deltas `[0, 29]`, a sample starting at frame `t` is supervised at `t+29`,
not `t+30`. At an episode boundary, RGB is clamped using the existing dataset
behavior; auxiliary label masks are false if the requested future frame was
outside the episode. All-padded batches have a graph-connected zero loss.

The first version supervises compact dynamic states rather than all 17 sidecar
features: Hanoi's large/medium/small disk world XYZ (9 scalars), Microwave's
food world XYZ and door hinge angle (4 scalars). Static entities, symmetric
disk rotations, extents and TCP-relative coordinates do not contribute to
this loss. Each head emits the schema-shaped tensor for compatibility, with
an explicit task-specific supervision mask saved in the config. The readout
sees only future DINO patches with spatial coordinates and learned entity
queries; it has no current-state or VLM bypass.

Use train-only mean/std from the existing geometry preflight, masked
Smooth-L1 loss, and initial weight 0.1. This is an initial hyperparameter,
not a demonstrated optimum. Training logs expose `loss_wm_geometry`,
`wm_position_rmse_m` (per-coordinate RMSE in meters) and
`wm_angle_mae_rad` (door-angle MAE in radians; zero when no angle is valid).

## Launch

```bash
conda activate lawam
cd /data/home/zyh/LaWAM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export LAWAM_ASSET_ROOT="$PWD"
export INTERACTION_DATA_ROOT=/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets
export INTERACTION_REVIEWED_ROOT="$PWD/results/interaction_sidecars_reviewed_20261025"
export INTERACTION_OUTPUT_ROOT="$PWD/results/Checkpoints/interaction_wm"
export INTERACTION_STEPS=40000
export INTERACTION_SEED=2026
export INTERACTION_AUX_WEIGHT=0.1
unset FINGERTIP_ROOT

# Each line is an independent 8-GPU job. Run one at a time on the same machine.
bash scripts/train_interaction_reviewed.sh hanoi visual_wm_aux
bash scripts/train_interaction_reviewed.sh hanoi visual_wm_aux_detach
bash scripts/train_interaction_reviewed.sh microwave visual_wm_aux
bash scripts/train_interaction_reviewed.sh microwave visual_wm_aux_detach
```

Append `--dry-run` to generate and validate config/data/assets without launching
training. H30, full Stage 2, 40k optimizer steps, 8 ranks x batch 4, accumulation
1, global batch 32, seed 2026, save every 10k plus final. Pretrain initialization:
`results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt`;
adapted LAM:
`latent_action_model/logs/dino_large_vae_dexjoco_bimanual_exclude_photograph_fingertip_stage1/pytorch_model.pt`.

Checkpoints are stored below
`results/Checkpoints/interaction_wm/<full_task_name>/<variant>/<run>/`.
For migration transfer the modified/new code, original datasets, converted
sidecars (including manifests and accepted episode lists), and model assets.
Set the three root variables to their destination locations and regenerate
configs. A clone alone does not carry uncommitted changes or ignored assets.

## Validation

```bash
PYTHONPATH=. python -m unittest scripts.test_wm_geometry scripts.test_interaction -q
```

Tests exercise the production backend with tiny VLM/LAM components: gradient
isolation, baseline policy initialization/loss/action equivalence, label-free
inference, sampled-frame endpoint alignment and padding, dynamic target masks,
all-padded loss, matched generated configs and invalid configuration rejection.
This does not substitute for a real distributed GPU training run.
