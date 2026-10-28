# Multi-object Interaction Experiments

The future-DINO auxiliary experiments `visual_wm_aux` and
`visual_wm_aux_detach` are documented in [wm_geometry_aux.md](wm_geometry_aux.md).

## Accepted reconstructed-label runs (2026-10-25)

The user explicitly accepted the supplied reconstructed poses after visual inspection.
This is a separate acceptance policy, not a relaxation of the original strict audit
flags or a claim of measured object ground truth. The converted manifest keeps each
episode's original audit, strict validity and source hash. Missing poses still fail
the data contract: Hanoi episodes 81/88 are omitted from all four variants.

Compatible files have been generated under
results/interaction_sidecars_reviewed_20261025/. Hanoi uses 98 episodes (97 train,
1 validation); microwave uses 100 (99 train, 1 validation). All four variants in
each task use the same episode list and geometry-manifest hash. The raw source
dataset remains the input; the delivered augmented Parquet is not required.

In the LaWAM environment, launch one job at a time per 8-GPU machine:

~~~bash
conda activate lawam
cd /data/home/zyh/LaWAM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
unset FINGERTIP_ROOT
bash scripts/train_interaction_reviewed.sh hanoi baseline
bash scripts/train_interaction_reviewed.sh hanoi visual
bash scripts/train_interaction_reviewed.sh hanoi visual_aux
bash scripts/train_interaction_reviewed.sh hanoi oracle
# Replace hanoi with microwave for that task's four runs.
~~~

Append --dry-run to check one launcher without training. The recipe remains
H30, full Stage 2, 40k updates, 8 GPUs, batch 4/rank, accumulation 1, seed 2026.
Checkpoints go under results/Checkpoints/interaction_reviewed/<task>/<variant>/;
save interval is 10k. Eight generated configs are also available under
results/interaction_reviewed_configs_20261025/. No 8-GPU training was launched
during preparation. Oracle deployment still requires the synchronized live
geometry evaluator/server described below.

For migration, copy the compatible sidecar directory including manifests, episode
lists and episode NPZ files, as well as the original task datasets and model assets.
Set INTERACTION_REVIEWED_ROOT, INTERACTION_DATA_ROOT and LAWAM_ASSET_ROOT
to the destination paths; regenerate configs through the launcher. Code changes
must be committed or otherwise transferred because a clone omits uncommitted files.

To recreate the compatible files from the DexJoCo agent's delivery (use a new output
directory if it already exists), run in the DexJoCo environment:

~~~bash
export PYTHONPATH="$PWD:/path/to/dexjoco/dexjoco:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
for task in hanoi microwave; do
  python scripts/prepare_interaction_candidates.py \
    --task "$task" \
    --source-root /path/to/multiobject_pose_20261024 \
    --data-root /path/to/dexjoco_lerobot_datasets \
    --output results/interaction_sidecars_reviewed_20261025 \
    --accept-reconstructed \
    --acceptance-note 'Reconstructed labels accepted after visual review for these experiments'
done
~~~

The converter uses saved current body poses, current door qpos, XML-fixed
interaction transforms, and the original row's current TCP positions. It checks
the original Parquet SHA256, frame mapping, geometry finiteness and rotation
contract; it never reruns physics or overwrites the supplied candidates. Offline
conversion is tested against live MuJoCo extraction for both tasks.

This experiment compares four matched H30 Stage 2 policies on Hanoi and microwave cook:

| Variant | Action conditioning | Geometry labels during training | At inference |
| --- | --- | --- | --- |
| `baseline` | Original LaWAM path | No | Existing LaWAM server |
| `oracle` | Current object geometry tokens | Yes | `scripts/serve_lawam_interaction.py` |
| `visual` | Current DINO patch features resampled to compact tokens | No | Existing LaWAM server |
| `visual_aux` | Same visual modules and token count as `visual` | Yes, auxiliary loss only | Existing LaWAM server |

All variants use the same pretrained checkpoint, data cohort, episode split, initialization of shared weights, optimizer recipe, 40,000 steps, 4 samples per GPU, 8 GPUs, gradient accumulation 1, action horizon 30, and global batch 32. The new DiT residual branches are zero-initialized. `visual_aux` has no inference-time dependency on geometry labels. By default, the H30 recipe uses the task-specific DexJoCo dataset and the matching LaWAM Stage 1 LAM checkpoint.

The template follows the existing Hanoi/Microwave C H30 full Stage 2 recipe: flow, perceptual, and latent distillation losses; trainable VLM and LAM decoder; action-expert-only mode disabled. Photograph's separate endpoint fingertip head is disabled by default because its labels are task-specific. To enable the same endpoint auxiliary loss in all four variants, set `FINGERTIP_ROOT` to valid per-task sidecars; preflight verifies every selected episode. Its weight is 0.1. Do not enable it for only one variant.

The default LAM is the existing `dino_large_vae_dexjoco_bimanual_exclude_photograph_fingertip_stage1` asset. It was adapted on tasks including Hanoi and microwave; it is not a target-excluded Stage 1 model for these two tasks. All four controls deliberately use this same asset. The official Stage 2 initialization is `results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt`, not a previously fine-tuned orange/green checkpoint.

## Dataset facts

The existing Hanoi and microwave LeRobot Parquet rows contain robot state and action but no per-frame object pose. Raw `replay.zarr` task state contains initial scene information, not a complete original simulator reset state or a sequence of original per-frame object poses. Replaying actions can produce candidate simulator reconstructions, but a successful replay is not proof of frame-by-frame object correspondence. As of 2026-10-24, episodes 0, 1, and 2 of each task failed the strict robot-state correspondence gate. Hanoi maximum absolute state discrepancies were 0.126, 0.234, and 0.321; microwave discrepancies were 0.706, 0.391, and 0.309. These are mixed-unit robot-state discrepancies, not object-position errors. The candidate manifests remain invalid and must not be used for training. This is a six-episode pilot, not an exhaustive replay audit. Details are in `results/interaction_sidecar_audit_20261024/report.json` in this workspace.

The preferred label source is a synchronous simulator recording made beside each demonstration image/state row using `starVLA.interaction_geometry.InteractionRecorder`. It records current geometry before the corresponding action. `scripts/export_interaction_recordings.py` checks every state, action, timestamp, and frame index against the LeRobot dataset before publishing an approved sidecar. For existing demonstrations, `scripts/export_interaction_replay.py` reconstructs candidates from raw action streams and `state[0]`; run it with `--preview`, inspect each replay against the original RGB trajectory, then explicitly approve only matching episodes with `scripts/approve_interaction_sidecars.py`. The approval command also requires the numerical robot-state alignment gate. It never marks failed candidates valid.

These reconstructed values must be described as simulator-derived labels, not measured object ground truth. Keep the audit output and approved episode list with the experiment. Do not reuse an approval list after changing the raw demos, task XML, simulator revision, or LeRobot rows.

## Prepare a common cohort

Place a reviewed `approved_episodes.json` in each task sidecar directory, and use that exact file for all four variants. The generator uses the same sorted episode set for training and validation, computes the geometry normalization on training episodes only, checks every sidecar row against the corresponding LeRobot state/action/timestamp, writes configs under `results/interaction_configs/`, and sends checkpoints under `results/Checkpoints/interaction/`.

For example, on a machine with both repositories and assets available:

```bash
export INTERACTION_DATA_ROOT=/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets
export LAWAM_ASSET_ROOT="$PWD"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8

export INTERACTION_SIDECAR_ROOT=/path/to/interaction_sidecars/bimanual_hanoi
export INTERACTION_EPISODES="$INTERACTION_SIDECAR_ROOT/approved_episodes.json"
```

Run the four Hanoi jobs. Set the task-specific cohort and sidecar paths again before microwave so that its runs use the reviewed microwave episodes.

```bash
bash scripts/train_interaction.sh hanoi baseline
bash scripts/train_interaction.sh hanoi oracle
bash scripts/train_interaction.sh hanoi visual
bash scripts/train_interaction.sh hanoi visual_aux
```

The script invokes the config generator and `train_lawam.sh`, which launches single-node Accelerate DDP with 8 ranks. For a preflight without launching training, append `--dry-run`. Oracle and Visual Aux require an approved `INTERACTION_SIDECAR_ROOT`. Baseline and Visual do not read the sidecar, but still take the same `INTERACTION_EPISODES` cohort for a valid comparison. Use separate task-specific environment values for the microwave commands. Do not launch any variant until its resolved config reports the same cohort hash as the other variants in that task.

```bash
export INTERACTION_SIDECAR_ROOT=/path/to/interaction_sidecars/bimanual_microwave_cook
export INTERACTION_EPISODES="$INTERACTION_SIDECAR_ROOT/approved_episodes.json"
bash scripts/train_interaction.sh microwave baseline
bash scripts/train_interaction.sh microwave oracle
bash scripts/train_interaction.sh microwave visual
bash scripts/train_interaction.sh microwave visual_aux
```

## Reconstruct candidate labels

Use the DexJoCo simulator environment, the unchanged LeRobot rows, and the corresponding raw demonstrations:

```bash
export PYTHONPATH="$PWD:/path/to/dexjoco/dexjoco${PYTHONPATH:+:$PYTHONPATH}"
python scripts/export_interaction_replay.py \
  --task hanoi \
  --data-root /mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets \
  --raw-root /mnt/ceph2/DexJoCo-Datasets-Raw/dexjoco_raw_datasets \
  --output /path/to/candidate_sidecars \
  --preview
```

Microwave uses `--task microwave`. The exporter writes to `OUTPUT/bimanual_hanoi` or `OUTPUT/bimanual_microwave_cook`, preserves a raw-frame map and reconstructed `qpos`, and rejects missing/ambiguous raw mappings. Review original and replay videos before making an episode list. For newly recorded simulator trajectories, call `InteractionRecorder.append(model, data, state=raw_state, action=raw_rotvec_action, frame_index=i, timestamp=i/30)` beside the image/state capture, then save one NPZ per episode and validate the directory with `scripts/export_interaction_recordings.py`.

After reviewing replay candidates, put only visually checked and numerically aligned episode IDs in a JSON array, for example `/tmp/hanoi_reviewed.json`, then record that decision with:

```bash
python scripts/approve_interaction_sidecars.py \
  --root /path/to/candidate_sidecars/bimanual_hanoi \
  --episodes /tmp/hanoi_reviewed.json \
  --note "Compared each replay with its source demonstration RGB"
```

This writes `approved_episodes.json` and marks unreviewed or numerically failing episodes invalid. Headless EGL previews require GPU/render-device access; rendering was verified outside the restricted sandbox in this workspace. Successful rendering does not imply successful trajectory alignment.

Neither export script can recover missing historical simulator state by itself. `export_interaction_recordings.py` only validates recordings already made with `InteractionRecorder`; it does not collect demonstrations. For guaranteed temporal pairing, record geometry synchronously with new demonstration RGB/state/action. If existing actions are replayed to create new demonstrations, save the newly rendered RGB and new robot state together with the geometry, convert that new dataset to LeRobot, and run all four variants on that same new dataset. Such recordings do not provide labels for the old RGB trajectories, and their demonstration quality still needs evaluation.

## Inference

Visual, Visual Aux, and Baseline use the original server/evaluator. Oracle needs geometry extracted from the live simulator in the same observation capture as RGB/state:

```bash
python scripts/eval_lawam_interaction.py ...
python scripts/serve_lawam_interaction.py ...
```

These CLIs retain the corresponding DexJoCo LaWAM CLI options. An Oracle checkpoint rejects a missing task/schema payload. `eval_lawam_interaction.py` provides the synchronized payload; a remote deployment must add an equivalent live MuJoCo extraction before calling the server. Do not provide reset-time geometry in place of current geometry.

## Migration and verification

Clone the code after committing the changed/new source files; a clone contains committed files only. This working tree also contains earlier orange/green changes, so keep their required source dependencies when packaging. No model weights, datasets, or sidecars are included by Git.

Copy the LeRobot task directories (including `meta`, `data`, and `videos`), approved sidecars and cohort JSON, and these assets to the destination:

- `results/Checkpoints/qwen3_weights`
- `results/Checkpoints/dinov3-vitb16-pretrain-lvd1689m`
- `results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt`
- `latent_action_model/logs/dino_large_vae_dexjoco_bimanual_exclude_photograph_fingertip_stage1/pytorch_model.pt`
- `latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml`

Set `LAWAM_ASSET_ROOT` to the directory with that layout. Alternatively set `LAWAM_PRETRAIN`, `LAWAM_LAM`, `LAWAM_VLM`, `LAWAM_DINO`, and `LAWAM_LAM_YAML` individually. The generator writes a local LAM YAML with the resolved DINO path, rather than retaining the source machine's absolute path. Regenerate configs on each machine and compare dataset cohort hashes and the initialization checkpoints. If copying symlinked assets/videos, dereference them or preserve their targets on the destination.

Pure model/data tests use the LaWAM environment:

```bash
PYTHONPATH=. python -m unittest scripts.test_interaction -q
```

Live simulator extraction/recording tests use the DexJoCo environment:

```bash
PYTHONPATH="$PWD:/path/to/dexjoco/dexjoco" python -m unittest scripts.test_interaction_simulator -q
```

This implementation passed 30 model/regression checks plus the real simulator test covering both tasks. A real Hanoi Baseline config/preflight also passed. No 8-GPU training run was started in this workspace. Default checkpoint frequency is 10,000 steps plus the final model, reducing disk use compared with the earlier 1,000-step orange saves.

## Geometry schema

`starVLA/interaction_geometry.py` defines fixed semantic slots and a per-feature validity mask. Hanoi has the base, three disks, and three posts. Microwave has the hot dog, microwave body, moving door, handle, start button, and interior. Every entity has world position, a 6D rotation representation, left/right hand-relative position, a joint-angle field, and an extent field (17 values). The door joint angle is current simulator state. DINO visual mode uses only the current-frame patch grid and produces the same number of compact tokens in `visual` and `visual_aux`.
