# Predicted-future compact visual tokens

Diagrams: [architecture (SVG)](figures/visual_pred_aux_architecture.svg) /
[PNG](figures/visual_pred_aux_architecture.png),
[training (SVG)](figures/visual_pred_aux_training.svg) /
[PNG](figures/visual_pred_aux_training.png),
[inference (SVG)](figures/visual_pred_aux_inference.svg) /
[PNG](figures/visual_pred_aux_inference.png).
Regenerate with `python docs/figures/draw_visual_pred_aux.py`.

`visual_pred_aux` follows `visual_aux`: the same compact-token encoder,
geometry readout and zero-initialized residual attention in the action expert.
The visual input changes from current DINO patches to the World Model's
**predicted** future DINO patches, and the auxiliary target changes from current
geometry to geometry at the corresponding future frame.

```text
current observations + instruction → VLM → predicted latent action
current DINO + predicted latent action → WM decoder → predicted future DINO
predicted future DINO + current VLM context + current robot state
    → 8 compact visual tokens
        → action expert interaction attention → actions
        → entity readout → future geometry loss (training only)
```

The default tokens are 8 × 128, with two query blocks and eight attention heads,
as in `visual_aux`. No future GT image, future GT robot state or geometry label
enters the token encoder. The existing inverse-dynamics attention path remains
active. Use the saved config for inference: although `visual_aux` and
`visual_pred_aux` have identical parameter keys, their visual inputs differ.

| Mode | Compact-token visual input | Geometry supervision | Extra action tokens |
| --- | --- | --- | --- |
| `visual_aux` | Current DINO | Current full geometry | Yes |
| `visual_wm_aux` | Separate future-patch readout | Future dynamic subset | No |
| `visual_pred_aux` | Predicted future DINO | Future full geometry | Yes |

## Targets and gradients

Reuse the existing audited interaction sidecars and train-only normalization.
Supervise all valid entries of the original 17-dimensional per-entity schema:
world XYZ, rotation 6D, left/right flange-relative XYZ, joint angle and extent.
This deliberately matches `visual_aux`; it does not use `visual_wm_aux`'s smaller
dynamic-state mask. Targets are normalized and reduced with masked Smooth-L1.
The default auxiliary weight is 0.1, logged as `loss_interaction`.

The loader uses the last actually sampled WM video frame, not a hardcoded
action-chunk endpoint. For video offsets `[0, 29]`, the target is at `t+29`.
Out-of-episode future requests produce false validity masks; an all-invalid
batch has a graph-connected zero geometry loss and still trains on the other
Stage 2 losses. `load_interaction_geometry=true` opens the sidecar store;
`load_future_interaction_geometry=true` selects the future targets. The
collator does not require current geometry in this mode.

The token encoder reads `h_t1_pred` without detaching it. Geometry gradients
reach the encoder, WM decoder and predicted latent branch. Action gradients
can also reach the WM through compact tokens after the zero-initialized
interaction residuals begin learning. The matched recipe retains
`detach_future_feature=true` for the **original direct flow-conditioning
path**; that setting does not detach the new token path. This gradient route
is an additional difference from `visual_wm_aux` and matters when interpreting
comparisons. Keep `future_prediction=true`, `flow_future_feature=predicted`,
`interaction_conditioning=true`, `action_expert_only=false`, and scheduled
sampling disabled; invalid combinations are rejected.

`visual_pred_interact_aux` and fingertip-object relation losses are deferred to
the second experiment. The default recipe has no fingertip loss.

## Generate configs and train

```bash
conda activate lawam
cd /data/home/zyh/LaWAM
export LAWAM_ASSET_ROOT="$PWD"
export INTERACTION_DATA_ROOT=/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets
export INTERACTION_REVIEWED_ROOT="$PWD/results/interaction_sidecars_reviewed_20261025"
export INTERACTION_OUTPUT_ROOT="$PWD/results/Checkpoints/interaction_pred"
export INTERACTION_AUX_WEIGHT=0.1
unset FINGERTIP_ROOT

# Validate data/assets and generate the resolved config without launching training.
bash scripts/train_interaction_reviewed.sh hanoi visual_pred_aux --dry-run
bash scripts/train_interaction_reviewed.sh microwave visual_pred_aux --dry-run

# Independent jobs; run one at a time on the same eight GPUs.
bash scripts/train_interaction_reviewed.sh hanoi visual_pred_aux
bash scripts/train_interaction_reviewed.sh microwave visual_pred_aux
```

The existing reviewed launcher explicitly selects the accepted reconstructed
sidecar cohort; those labels are not newly measured simulator ground truth.
To use a different approved GT sidecar store, use `train_interaction.sh` with
`INTERACTION_SIDECAR_ROOT` and the matching `INTERACTION_EPISODES` list.

Recipe defaults remain full Stage 2, H30, 40k steps, seed 2026, eight GPUs ×
four samples, accumulation one. `--variant visual_pred_aux` is also supported
by `scripts/configure_interaction.py`; `--variant all` retains the original
four-mode cohort generation behavior.

## Validation

```bash
PYTHONPATH=. python -m unittest scripts.test_visual_pred_aux scripts.test_wm_geometry scripts.test_interaction -q
```

Tiny production-backend tests cover matching architecture/initialization,
predicted-vs-GT input isolation, full future geometry including flange-relative
coordinates, gradient flow, masked endpoints, both action-expert architectures,
repeated diffusion batches, classifier-free guidance, label-free inference,
generated configs and invalid settings. These checks do not establish task
performance or substitute for distributed GPU training.
