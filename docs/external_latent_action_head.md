# External Latent Action Head

Set `framework.action_model.latent_action_head_type: external_qformer` to
predict the LAM latent outside the VLM. The default `vlm_queries` preserves
the original policy architecture.

The external head takes the entire VLM `last_hidden_state` sequence and its
attention mask. It uses one learned query, one cross-attention block, an FFN,
and LayerNorm to produce `[B, 1, lam.code_dim]`. For the released LAM,
`lam.code_dim` is 32. Its context is the same sequence the flow head receives,
before the flow head's own feature projection. This includes current primary
images, current wrist images, task text, and flow queries. Padded tokens are
masked out. Future expert images remain teacher/loss inputs only.

In this mode there are no latent action query tokens inside the VLM. The
remaining flow queries are unchanged. Training collation, inference batch
construction, and checkpoint validation all support this zero-query case.
`num_action_queries` configures the original mode only; the external head's
single query lives in the head itself.

The predicted latent is fed to the existing LAM decoder with current DINO
features. Its predicted future features condition the flow action head.
The external head is registered under the existing `policy_backend.vlm_to_lam`
parameter prefix to preserve pretrained initialization and optimizer settings.

## Initialization

The head has the same 19 tensor names and shapes as the released QFormer.
Use the official LaWAM pretraining checkpoint through
`trainer.pretrained_checkpoint` to reuse all 19, including the learned query,
attention projections, FFN, and normalization parameters. This is a warm
start: the input distribution changes from eight action-query outputs to
the full multimodal sequence, so the head still needs Stage 2 training.
The LAM encoder cannot initialize this head directly because its architecture
and inputs differ.

The provided experiment config loads the C-group Stage 1 LAM separately and
sets `trainer.load_pretrained_policy_lam: false`, preventing the official
Stage 2 initialization checkpoint from overwriting it. The VLM and compatible
flow parameters still load from the official checkpoint. No target-task
Stage 2 checkpoint is used for initialization.

## Training

Use `starVLA/config/training/starvla_train_dexjoco_bimanual_photograph_external_latent.yaml`
for the initial single-task comparison with C. It uses the same 40,000 steps,
4 samples per GPU, 5,000-step checkpoint interval, and loss weights:

```text
loss = loss_flow + 0.1 * loss_perceptual + 0.1 * loss_distill
```

Fingertip supervision is inherited through the C-group Stage 1 LAM; no
additional Stage 2 fingertip loss is enabled. The head learning rate is
`5e-5`, the same as the previous QFormer's base learning rate.

The head input is not detached: distillation and perceptual losses train the
head and backpropagate into the VLM. `detach_future_feature: true` preserves
the previous behavior in which flow loss does not backpropagate through the
predicted-future-feature branch. Structural separation alone does not isolate
gradients in the shared VLM. `flow_only_mode` requires the old VLM action-query
boundary and is rejected with the external head.

```bash
conda activate lawam
cd /data/home/zyh/LaWAM

HF_HOME=/tmp/lawam_hf_cache \
HF_HUB_CACHE=/tmp/lawam_hf_cache/hub \
HF_DATASETS_CACHE=/tmp/lawam_hf_cache/datasets \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_PROCESSES=8 \
bash train_lawam.sh \
  --config_yaml starVLA/config/training/starvla_train_dexjoco_bimanual_photograph_external_latent.yaml
```

To rerun a control with the same code and initialization, use this config and
override `--framework.action_model.latent_action_head_type vlm_queries`, with
a separate `--run_id` and `--run_root_dir`. The existing action-projection
expansion in the workspace was preserved; use the same initialization code
for both runs rather than assuming a historical C checkpoint is an exact
control. The external head additionally sees wrist features, so an improvement
alone does not isolate architecture separation from added visual information.

Saved Stage 2 configs record the mode. Serve a resulting checkpoint with the
existing LaWAM server; no additional inference flag or separately served model
is needed. The C-group exported LAM and the local VLM/DINO weight paths must be
available when the framework is constructed.

## Multi-task Experiment

Use `starVLA/config/training/starvla_train_dexjoco_bimanual_external_latent.yaml`
to train on all five bimanual tasks. Initialization, losses, learning rates,
and freeze settings match the external-head single-task config. The budget is
60,000 steps, matching the earlier C-group multi-task run, with global batch
32 on eight GPUs and checkpoints every 5,000 steps. It starts from the
official initialization and C-group Stage 1 LAM, independently of the ongoing
single-task run. Its output root and run ID are separate.

Each task has 100 episodes; the last episode is held out for validation.
There are 495 training episodes and 301,719 training frames in total. The
existing `balance_dataset_weights: true` multiplies equal mixture weights by
frame counts, so it samples by frames rather than equally by task:

| Task | Training Frames | Sampling Probability |
| --- | ---: | ---: |
| bimanual_hanoi | 106,222 | 35.21% |
| bimanual_photograph | 33,950 | 11.25% |
| bimanual_assembly | 53,705 | 17.80% |
| bimanual_microwave_cook | 67,645 | 22.42% |
| bimanual_unlock_ipad | 40,197 | 13.32% |

At 60,000 steps the policy draws 1.92 million training samples, equivalent
in volume to about 6.36 passes through the training frames. Sampling is with
replacement; this is not a guarantee that every frame is visited. Multi-task
training increases input diversity, while the number of updates to the head
depends on optimizer steps. It does not guarantee better target-task success.

```bash
conda activate lawam
cd /data/home/zyh/LaWAM

HF_HOME=/tmp/lawam_hf_cache \
HF_HUB_CACHE=/tmp/lawam_hf_cache/hub \
HF_DATASETS_CACHE=/tmp/lawam_hf_cache/datasets \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_PROCESSES=8 \
bash train_lawam.sh \
  --config_yaml starVLA/config/training/starvla_train_dexjoco_bimanual_external_latent.yaml
```

Run in parallel on a separate eight-GPU allocation or server. Compare the
external-head multi-task model with an original-head multi-task control at
the same 60,000 steps, dataset mixture, initialization, and evaluation seeds.
For a fresh control, use the multi-task config with
`--framework.action_model.latent_action_head_type vlm_queries`,
`--run_id dexjoco_bimanual_all_C_vlm_queries_control`, and
`--run_root_dir results/Checkpoints/dexjoco_bimanual_all_C_vlm_queries_control`.
Report per-task success and the unweighted mean across the five tasks.
Since all five tasks enter Stage 2 training, their evaluation measures
multi-task performance, not generalization to unseen tasks. The additional
wrist-feature access and action-projection initialization caveats above
also apply to this comparison.

## Verification

Nine CPU regression checks passed, including training/backpropagation,
inference, padding masks, and serving checkpoint reload. The real local Qwen3
processor also produced identical train/inference tokens, pixels, and masks
for the same prepared observations. All 19 head tensors were loaded from the
official checkpoint and compared exactly; the legacy mode matched the old
QFormer output exactly. Full GPU training and simulator evaluation have not
been run for this change.

Run CPU regression checks without loading the large VLM or using GPUs:

```bash
CUDA_VISIBLE_DEVICES='' NO_ALBUMENTATIONS_UPDATE=1 \
MPLCONFIGDIR=/tmp/mpl_lawam XDG_CACHE_HOME=/tmp/lawam_cpu_checks \
python -m unittest discover -s scripts -p 'test_external_latent_action_head.py' -v
```
