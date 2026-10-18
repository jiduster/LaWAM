# Teacher Latent Diagnostic

This is an offline, privileged-future diagnostic, not a training change or a
deployable evaluation policy. It loads one Stage 2 checkpoint without updating
any weights. It supports both `vlm_queries` and `external_qformer` checkpoints.

## Conditions

| Name | Future feature given to the action expert |
| --- | --- |
| `predicted` | Current images/text -> VLM latent -> checkpoint WM decoder |
| `teacher_latent` | Current/future expert images -> checkpoint IDM latent -> same WM decoder |
| `gt_future` | Real future expert image -> frozen DINO feature extractor |

Current VLM features, state, flow weights, guidance, and inference steps are
unchanged. Within each repeat, the three conditions share identical initial
flow noise. Errors are averaged over repeats, rather than evaluating a
best-of-N prediction or comparing a mean action to the demonstration.

The IDM is the encoder in the loaded Stage 2 checkpoint, not a separate release
or Stage 1 checkpoint. It uses the same preprocessing and temporal pair as the
checkpoint's distillation target. No future image is added to the VLM prompt.

## Run

Activate the `lawam` environment and run from the repository root in a terminal
with an allocated GPU. On Pair Slurm, select a free physical GPU in Monica and
allocate it with `srun --gres=gpu:INDEX --pty bash`; inside that allocation use
`cuda:0`, not the physical GPU index. Do not nest `srun` in an existing job.

```bash
source /data/home/zyh/miniconda3/bin/activate
conda activate lawam
cd /data/home/zyh/LaWAM

PYTHONPATH=/data/home/zyh/LaWAM \
HF_HOME=/tmp/lawam_hf_cache \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
NO_ALBUMENTATIONS_UPDATE=1 \
MPLCONFIGDIR=/tmp/mpl_lawam \
OMP_NUM_THREADS=1 \
python scripts/diagnose_lawam_teacher_latent.py \
  --checkpoint results/Checkpoints/dexjoco_bimanual_photograph_C_h30/20261015_001205+dexjoco_bimanual_photograph_C_h30_sft/final_model \
  --dataset bimanual_photograph \
  --split val \
  --windows-per-episode 64 \
  --batch-size 4 \
  --num-workers 0 \
  --noise-repeats 3 \
  --seed 2026 \
  --device cuda:0 \
  --output-dir /tmp/lawam_teacher_C_h30
```

The checkpoint argument accepts a `.pt` file, `final_model`, or a run directory.
The output directory must not contain a previous diagnostic. Add
`--max-batches 1 --noise-repeats 1` for a short model smoke test, using a new
output directory. Add `--dry-run-data` to validate data/collation on CPU without
loading model weights. `--data-root` relocates the LeRobot data root.

`--prefix-steps N` additionally measures the first N actions. Set N to the
actual execution/replanning length when that differs from the training chunk.
Inference steps default to the checkpoint setting. Guidance defaults to 1.0;
set `--guidance-scale` explicitly if the actual policy uses another value.

## Data And Metrics

The script reuses the saved split, training normalization statistics, LeRobot
reader, and evaluation collator. It does not change `val_tail_ratio`. Explicit
episode IDs via `--episodes` must already belong to the chosen split. Selecting
`--split train` is allowed for a training-versus-validation comparison, but its
report is marked as not held out. If training used `train_split_all=true`, the
validation split is also marked as not held out from Stage 2.

Windows are uniformly spaced within each episode, sampled without replacement,
and restricted to complete video/action windows. Episode-tail replication is
excluded. The current photograph run has just one held-out episode, so 64
windows are not 64 independent demonstrations. More held-out episodes require
new data, or a new training run that reserves them from the outset.

Artifacts:

- `manifest.json`: checkpoint, settings, episode IDs, frame pairs, and warnings.
- `per_window.csv`: latent, future-feature, and paired action metrics.
- `summary.json`: window means, episode macro means, per-episode results, and
  paired action-MSE reductions. Positive reduction means the oracle improved
  on the normal prediction.

`latent_mse` and `latent_cosine` compare predicted and teacher latents.
`future_mse/predicted`, `/teacher_latent`, and `/current` compare the respective
features with the real future DINO features. `/gt_future` is zero by definition.

`action_mse/MODE/all` and `action_mae/MODE` use checkpoint-normalized action
units, without clipping. For 44D DexJoCo bimanual actions, arm and hand MSE are
also reported separately: arm indices `[0:6,22:28]`, hand indices `[6:22,28:44]`.
`action_change_mse/MODE` measures action sensitivity to the replacement, using
the paired normal prediction as reference. These are not physical position
errors, FK errors, flow-training losses, or closed-loop success rates.

## Interpretation

- Better `teacher_latent` future features AND actions support investigating the
  deployable latent predictor as a bottleneck.
- Better `gt_future` but little `teacher_latent` improvement motivates examining
  the latent bottleneck/decoder as well as the predictor.
- Better future features without better actions motivates examining the
  action expert's use of future information.
- A null/worse oracle result is not definitive: oracle conditions can differ
  from the predicted features the action expert saw during training, and
  offline imitation errors do not fully determine closed-loop success.

Even a positive teacher result does not establish that current observations
contain enough information to predict that future. The teacher sees privileged
information. Human pretraining is one hypothesis to test, not a guaranteed way
to realize an oracle improvement.

## Regression Tests

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=. \
python -m unittest scripts.test_lawam_teacher_diagnostic -v
```

Tests cover both latent-head architectures, agreement with normal inference,
future-image isolation from the normal branch, paired noise/RNG restoration,
action masks, full-window selection, and episode-level aggregation.
