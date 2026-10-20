# Oracle Shutter Contact Experiment

This is a privileged-information experiment for `bimanual_photograph`. It
conditions actions on **current** simulator geometry, not predicted future
fingertips. It does not add a geometry prediction loss or modify Stage 1.

## Model

```
images + prompt -> VLM -----------------------> semantic attention
                    | latent
current DINO ------> LaWM future DINO --------> visual attention
                                                        |
current MuJoCo state -> contact encoder -> contact attention
                                                        |
robot state + noisy action chunk --------------------> action expert -> actions
```

The contact encoder produces 17 tokens:

- One shutter target token: world XYZ and outward unit normal.
- Eight fingertip tokens: world XYZ; left FF/MF/RF/TH, then right FF/MF/RF/TH.
- Eight relationship tokens: target-minus-tip vector, Euclidean distance,
  and signed displacement along the target normal.

All positions are in metres in the MuJoCo world frame. Positions/distances
are divided by 0.1 inside the encoder. Learned identity embeddings distinguish
the token roles and fingers. No action normalization is applied to this input.

In the 16-layer action expert, a dedicated residual cross-attention follows
blocks 4, 8, 12, 16. Action hidden states query contact tokens as keys/values.
Its output projection starts at zero, preserving the pretrained policy's
initial outputs. BC gradients train both the attention and contact encoder.
At the first optimizer step, the zero output projection gates upstream
gradients; subsequent steps train the encoder too. There is no new loss weight.

Contact conditions remain present in both CFG branches, as does current
proprioception; the existing CFG mechanism drops future vision only.

Depth queries remain a separate VLM feature. The supplied contact config
disables depth to isolate this experiment. Combining them requires enabling
depth and supplying its original targets/settings; it is a different experiment.

## Physical Definitions

`starVLA/contact_geometry.py` is the single definition used offline and online.

- Target: outer face centre of `camera_body_top_cap`, a thin cylinder. Its
  outward direction is local **-Z**, corresponding to camera-body +Y in the
  current XML. The point is geom centre plus outward normal times half-height.
- Fingertips: centres of collision capsules attached to `ff_tip_*`, `mf_tip_*`,
  `rf_tip_*`, `th_tip_*`. These are contact-pad reference points, not actual
  contact locations and not the distal joint origins from older sidecars.
- Input tensor: `[9,6]`; target uses XYZ+normal, fingers use XYZ+three zeros.

This is an upper-bound test for access to shutter contact geometry. It does
not supply every task variable: logo alignment and the full camera orientation
still depend on vision. A null result would not rule out all oracle object
state designs. Simulator GT is required at inference; this is not yet a
deployable real-robot perception solution.

## Data Prerequisite

The inspected LeRobot photograph dataset contains robot `observation.state[46]`
and `action[44]`, but **no per-frame object pose or full qpos**. Simulator
`camera_ori_pose` is a reset-time value and cannot substitute for current pose.
Existing fingertip sidecars also cannot recover a moving object's trajectory.
Do not start oracle training until aligned contact sidecars have been produced.

Two supported sources:

1. During demonstration collection, instantiate `PhotographContactRecorder`
   per episode. Call `append(model, data, raw_state46, frame_index, timestamp)`
   at exactly the time each corresponding image/state is saved, before the
   next action. Call `save(path_to_episode_npz)` when done. Frame indices start
   at zero; timestamps are episode-relative, matching the LeRobot rows.
2. Existing full simulator recordings can be converted without rendering.
   Each `episode-NNNNNN.npz` must contain `qpos[T,nq]`, `state[T,46]`,
   `frame_index[T]`, `timestamp[T]`, together with `episode-NNNNNN.xml` for that
   episode's model (including reset-time scene modifications, with assets
   resolvable). Reconstructed robot states are checked against the recording.

```bash
# Run only if full simulator records are available; substitute real directories.
conda activate dexjoco
PYTHONPATH=/data/home/zyh/LaWAM python scripts/record_photograph_contact_from_qpos.py \
  --raw-root /path/to/full_simulator_records \
  --output-root /path/to/aligned_contact_recordings

conda activate lawam
PYTHONPATH=/data/home/zyh/LaWAM python scripts/prepare_photograph_contact.py \
  --dataset-root /mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets/bimanual_photograph \
  --recordings /path/to/aligned_contact_recordings \
  --output-root /mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_contact_sidecars
```

The preparation command checks every episode's frame count, indices, timestamps
and raw 46D state before creating sidecars. A manifest records parquet SHA256
hashes. The training launcher revalidates them before starting eight workers.
This detects wrong robot trajectories/alignment, but cannot certify that an
arbitrary external object's labels are genuine GT; that requires faithful
simulator recording and the correct per-episode model.

Sidecar layout is `<root>/bimanual_photograph/manifest.json` and
`episodes/episode-NNNNNN.npz`. The collator reads only the current frame.
Missing/invalid sidecars cause errors, never zero-filled fallback conditions.

## Training

Both groups start from the same existing C_h30 Stage 2 checkpoint and train
10,000 additional optimizer steps with the same data split and existing-module
learning rates. New contact modules use 1e-4; existing flow and LaWM decoder use
1e-5; VLM uses 2e-6. This is weight initialization with a new optimizer/schedule,
not resumption of the previous optimizer state.

```bash
source /data/home/zyh/miniconda3/bin/activate
conda activate lawam
cd /data/home/zyh/LaWAM

# Requires prepared sidecars. Runs on allocated GPUs visible to this shell.
NUM_PROCESSES=8 bash scripts/train_oracle_contact.sh oracle

# Separate job/server, matched 10k continuation without contact conditioning.
NUM_PROCESSES=8 bash scripts/train_oracle_contact.sh control
```

Override paths using `CONTACT_SIDECAR_ROOT` and `CONTACT_DATA_ROOT`. Additional
CLI training overrides are forwarded to `train_lawam.sh`. Output roots are
separate for oracle/control. Use identical evaluation seeds and rollout counts.

## Inference

Use the new wrappers; the old DexJoCo server discards additional observation
keys and the old evaluator does not emit current contact geometry.

```bash
conda activate lawam
cd /data/home/zyh/LaWAM
PYTHONPATH=/data/home/zyh/dexjoco/dexjoco:/data/home/zyh/LaWAM \
python scripts/serve_lawam_contact.py \
  --checkpoint /path/to/contact_run/final_model/pytorch_model.pt \
  --lawam-root /data/home/zyh/LaWAM --host 127.0.0.1 --port 8000 \
  --robot-type dual_arm --single-arm-stats-key dexjoco_bimanual \
  --dual-arm-stats-key dexjoco_bimanual --action-hz 30 --log-actions-every-n 20

# In the environment used for your existing DexJoCo evaluations:
PYTHONPATH=/data/home/zyh/dexjoco/dexjoco:/data/home/zyh/LaWAM \
python scripts/eval_lawam_contact.py \
  --config /data/home/zyh/dexjoco/configs/multi_task/bimanual_photograph.yaml \
  --host 127.0.0.1 --port 8000 --seed 0 --episodes 50 --action-horizon 30
```

The evaluator samples geometry with each current image and passes the snapshot
to the existing inference worker. The server applies existing robot/action
normalization unchanged, forwarding contact geometry separately. Checkpoint
loading rejects missing or unexpected contact weights relative to the saved
config. The external DexJoCo checkout is not modified.
For the control checkpoint, use the original server and evaluator. The new
server explicitly rejects a checkpoint with contact conditioning disabled.

## Verification

CPU tests cover baseline equivalence at initialization, both DiT variants,
CFG, repeated flow batches, BC gradient flow, action sensitivity, strict state
reload, current-only sidecar access, missing input errors, dataset alignment,
checkpoint/config mismatches and optimizer learning rates. The real photograph
XML extraction can be checked without rendering using the MuJoCo test script.
No full training or rollout is implied by these unit tests.
