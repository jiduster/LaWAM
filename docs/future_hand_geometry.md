# Future Hand Geometry Conditioning

This Stage 2 experiment predicts eight Allegro fingertips at multiple times
and feeds the predictions directly into the flow action expert. Existing
configurations leave the new modules disabled.

## Architecture

1. The existing VLM latent and LAM decoder predict future DINO tokens.
2. A three-layer MLP reads their spatial mean plus the masked, normalized
   current proprioceptive state. It predicts `[B,3,8,3]` normalized coordinates.
3. Each coordinate is encoded with a coordinate MLP, a physical-time embedding,
   a left/right embedding, and a finger embedding, yielding 24 geometry tokens.
4. Separate geometry cross-attention residuals follow DiT blocks 4, 8, 12, 16.
   They run on every flow update, including inference. They do not append tokens
   to the existing vision/semantic streams or generate a second action chunk.

The configured offsets `[6,15,29]` are zero-based action-frame offsets from the
current observation: 0.2, 0.5, and 0.9667 seconds at 30 Hz. A single MLP predicts
all three times from the predicted endpoint feature; this version does not roll
out intermediate DINO images autoregressively.

The targets are absolute fixed-ego-camera optical coordinates in meters, using
the existing sidecars. They retain whole-hand motion; this first version does
not introduce wrist-local coordinates, object/contact labels, or a new FK loss.
Tip order is `left_ff,left_mf,left_rf,left_th,right_ff,right_mf,right_rf,right_th`.
`ff/mf/rf/th` mean index/middle/ring/thumb, respectively.

Only predicted geometry reaches action conditioning, in both training and
inference. Future images enter the existing teacher/perceptual losses; future
coordinates enter only the geometry loss. Inference needs no sidecars.

## Loss And Initialization

The default objective is:

```text
L = L_flow + 0.1 L_perceptual + 0.1 L_distill + 0.05 L_geometry
```

Geometry uses elementwise Smooth L1 on `(position - [0,0,1.3]) / 0.1`, averaged
over valid times, tips, and coordinates. Nonfinite labels and episode-tail
padding are masked. An all-invalid batch contributes zero geometry loss.
`geometry_rmse_mm` is coordinate RMSE in millimeters, not mean Euclidean tip
distance. `geometry_valid_fraction` reports the fraction of valid points.
All three metrics are logged for both training and validation.

The new MLP/token encoders and attention Q/K/V are randomly initialized. Each
geometry attention output projection (weight and bias) starts at zero, making
the new residual initially an identity. BC immediately updates these output
projections; gradients to upstream token encoders start after they become
nonzero. The geometry prediction MLP receives supervised gradients immediately.

The original future-DINO-to-flow branch retains `detach_future_feature=true`.
The new geometry branch deliberately carries both geometry-supervision and BC
gradients into the WM decoder and predicted latent. CFG drops geometry together
with the future visual condition; the unconditional branch receives zero tokens.

The supplied config starts from the official full Stage 2 pretraining checkpoint
and C's Stage 1 checkpoint that excludes photograph. It does not continue an
already trained photograph Stage 2 checkpoint. `load_pretrained_policy_lam=false`
prevents the official full checkpoint from overwriting the selected Stage 1 LAM.
Original flow weights load via the existing action-dimension expansion. New
geometry keys are expected to be reported as newly initialized at startup.

## Train

Run in the `lawam` environment from the repository root in an existing allocation
with eight GPUs. Preserve the scheduler's `CUDA_VISIBLE_DEVICES`. On Pair Slurm,
choose indices using Monica, and allocate with one `gpu:I` token per card; never
nest `srun` inside an existing job.

```bash
source /data/home/zyh/miniconda3/bin/activate
conda activate lawam
cd /data/home/zyh/LaWAM

HF_HOME=/tmp/lawam_hf_cache \
HF_HUB_CACHE=/tmp/lawam_hf_cache/hub \
HF_DATASETS_CACHE=/tmp/lawam_hf_cache/datasets \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/mpl_lawam OMP_NUM_THREADS=1 \
NUM_PROCESSES=8 \
bash train_lawam.sh \
  --config_yaml starVLA/config/training/starvla_train_dexjoco_bimanual_photograph_C_h30_geometry.yaml \
  --run_id dexjoco_bimanual_photograph_C_h30_geometry_sft
```

This uses 99 training episodes, one validation episode, batch size 4 per GPU
(global batch 32), horizon 30 at 30 Hz, 40,000 steps, and saves every 5,000 steps.
Outputs are under
`results/Checkpoints/dexjoco_bimanual_photograph_C_h30_geometry/<timestamp>+<run_id>`.

The config expects existing fingertip sidecars at
`/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_fingertip_sidecars`.
No labels need regeneration for photograph. New tasks require aligned sidecars
in the same coordinate frame and tip order.

For an auxiliary-only ablation, use a distinct run ID and append:

```text
--framework.action_model.geometry_inject false
```

This retains identical multi-time geometry supervision without token encoders
or geometry attention. For a baseline, append both:

```text
--framework.action_model.geometry_conditioning false
--framework.action_model.geometry_loss_weight 0.0
```

Use the same initialization, training budget, and evaluation seeds for comparisons.
To test whether future geometry adds value beyond current geometry, reserve a
separate matched experiment; that control is not implemented by these switches.
After training, the saved config reconstructs the geometry modules via the
existing policy server entrypoint. Deploy the code changes to the serving machine.

## Verification

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 \
NO_ALBUMENTATIONS_UPDATE=1 MPLCONFIGDIR=/tmp/mpl_lawam \
python -m unittest scripts.test_future_hand_geometry \
  scripts.test_external_latent_action_head scripts.test_lawam_teacher_diagnostic -v
```

Tests cover frame selection, masking/nonfinite labels, zero-initialized policy
equivalence, BC gradients through geometry into WM, action sensitivity, future
label isolation, CFG, both DiT variants, optimizer grouping, initialization,
checkpoint reload, and compatibility with offline teacher diagnostics.
