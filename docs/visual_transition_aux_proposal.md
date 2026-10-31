# Joint current / predicted-future visual tokens (proposal)

Status: architecture proposal only; no policy mode or training code is added.
Suggested experimental name: `visual_transition_aux`.

## Motivation and verified starting point

The action head in `flowmatching_expert.py` concatenates current patches,
predicted future patches and projected VLM tokens along the sequence axis.
The default AlternateVLDiT alternates visual/text cross-attention using masks.
`visual_aux` and `visual_pred_aux` independently compress one of the two visual
streams. The proposed branch compresses their ordered pair into eight shared
tokens for the existing interaction adapter. This provides an inductive bias
towards representing transitions, not a guarantee of learning inverse dynamics.

The user's reported improvements of each single-stream variant motivate this
test; they do not establish that the streams offer complementary policy gains.
The predicted future is generated from the current feature and policy-predicted
latent, not an independent observation of the future.

## First experiment: shared queries over two-frame memory

For current features H0 and predicted future features H1_hat, both [B,K,D]:

```text
X0 = W(H0)     + spatial_xy + learned_time_current
X1 = W(H1_hat) + spatial_xy + learned_time_future
M  = concat(X0, X1, dim=tokens)                       # [B,2K,128]
Q  = learned_queries + pooled_current_VLM + current_state_condition
Z  = LayerNorm(QueryBlock_2(QueryBlock_1(Q, M), M))    # [B,8,128]
```

Share the visual projection and spatial MLP across the two streams; keep the
eight queries, 128-dimensional output and two QueryBlocks from visual_aux.
Use the actual K; do not assume a particular DINO grid resolution. Source/time
embeddings identify observed-current versus predicted-future features. For a
variable prediction interval, also condition the future stream on the interval.

Do not reshape 2K as a single square image: construct each K-patch grid's XY
coordinates first, then concatenate. Do not interpret same-index feature
subtraction as object motion: these are image locations, not tracked entities.

The final eight slots are jointly inferred from both streams. Concatenating two
independent eight-slot encoders into sixteen slots is a useful late-fusion
control, but changes the downstream token budget and is not the primary design.

## Supervision and gradients

Read both endpoints from the SAME Z, using entity queries plus a current/future
readout-time embedding and shared attention/projection weights. Targets never
enter the encoder. The endpoint geometry losses use the existing 17-feature
schema, per-feature masks and training-only normalization.

```text
G0_hat = readout(Z, entity_queries + readout_time_current)
G1_hat = readout(Z, entity_queries + readout_time_future)
L_aux  = lambda_geo * (L_geo_current + L_geo_future) / 2
```

Start with lambda_geo=0.1 to match the previous nominal auxiliary budget;
the gradient magnitudes still need monitoring. This is a proposed initial
setting, not a measured optimum. Each endpoint is reduced over its own valid
features. All-invalid endpoints produce graph-connected zero losses.

Keep the original dense visual action-conditioning path for the first
experiment. Match visual_pred_aux's gradient setting: predicted patches are
attached in the compact branch; the original direct future-feature path retains
its existing detach setting. Geometry gradients and action gradients can reach
the WM through compact tokens (the action residual is initially zero).

Inference uses only current views, language, robot state and model predictions.
No GT future image/state/geometry is supplied. The existing perceptual and
latent-distillation losses continue during training.

## Optional second experiment: explicit geometric transition

Only after the basic joint model, add a small delta readout from Z for dynamic
entities and physically meaningful components:

```text
delta_p = object_world_xyz_future - object_world_xyz_current
delta_r = (object_xyz - flange_xyz)_future - (object_xyz - flange_xyz)_current
delta_theta = door_angle_future - door_angle_current
valid_pair = valid_current AND valid_future
```

Initial targets: the three moving disks for Hanoi; food translation and door
hinge motion for Microwave. Relative-to-flange changes are optional. Use the
joint's actual range/wrap convention for angle changes, and train-only delta
scales with a floor so near-static dimensions are not excessively amplified.
Do not subtract 6D rotation vectors and call the result an angular displacement;
rotation-change supervision would require relative rotations and symmetry care.

These labels are derived from existing geometry and add no new annotation
information. Their proposed benefit is explicit weighting of change. A matched
zero-motion baseline, moving-phase metrics and loss-weight controls are needed.

## If current information gets diluted

An alternative is current-anchored fusion: obtain Z0 with the current encoder;
let Z0 query the future patches and add a learned gated residual to produce Z.
This is a separate capacity/compute change. A small initially positive gate is
an experimental option, not a calibrated confidence score. Compare it with an
equally sized current-current adapter. Two endpoints alone do not identify a
unique action trajectory or contact sequence.

## Minimal ablations

1. Existing visual_aux / visual_pred_aux on the same cohort and evaluation seeds.
2. Joint memory + current-only geometry: match visual_aux supervision while
   changing compact memory. Keep the same two-readout architecture in controls,
   using a graph-connected zero for an inactive head.
3. Joint memory + averaged current/future geometry: separate dual supervision.
4. Same joint architecture with H0 in both memory slots, keeping future labels:
   parameter/token/target-matched control for access to predicted future.
5. Optional dynamic delta target, with separately reported compute and weights.

Keep the original dense path identical across these runs. The current-current
control tests only the compact branch; the policy still receives future features
through the original path. To test a full replacement, train a separate model
without the dense visual path after joint fusion demonstrates value.

At evaluation, replace/shuffle only the compact branch's future memory while
holding other paths fixed. This is a dependence diagnostic with distribution
shift, not proof of useful dynamics reasoning. Compare with the trained
current-current control and success rate on matched starts/seeds. Endpoint
decoding and attention weights alone do not establish transition understanding.

## Figures

- `figures/visual_transition_aux_architecture.svg` / `.png`
- `figures/visual_transition_aux_training.svg` / `.png`

Regenerate with `python docs/figures/draw_visual_transition_aux.py`.

## Implementation references

- `starVLA/model/framework/vlas/flowmatching_expert.py`: token-axis concatenation
  of current/future/VLM features in training and inference.
- `starVLA/model/framework/vlas/cross_attention_dit.py`: original attention and
  the separate compact-token interaction adapters.
- `starVLA/model/framework/vlas/interaction_tokens.py`: current single-frame
  resampler, entity readout and masked geometry loss to extend in a later task.
