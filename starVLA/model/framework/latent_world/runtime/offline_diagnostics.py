"""Paired, privileged-future diagnostics. Never used by the serving path."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from starVLA.model.framework.vlas.lawam import LatentWorldPolicyBackend


CONDITIONS = ("predicted", "teacher_latent", "gt_future")


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _mse_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {tuple(prediction.shape)} != {tuple(target.shape)}")
    return (prediction.float() - target.float()).square().flatten(1).mean(dim=1)


def masked_error_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    absolute: bool = False,
) -> torch.Tensor:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("Prediction, target, and action mask must have identical shapes.")
    valid = mask.bool()
    count = valid.flatten(1).sum(dim=1)
    if (count == 0).any():
        raise ValueError("Every evaluated sample must contain valid actions.")
    error = prediction.float() - target.float()
    error = error.abs() if absolute else error.square()
    return error.masked_fill(~valid, 0).flatten(1).sum(dim=1) / count


@torch.inference_mode()
def diagnose_train_batch(
    backend: "LatentWorldPolicyBackend",
    batch: dict,
    *,
    noise_repeats: int = 3,
    noise_seed: int = 2026,
    num_inference_steps: int | None = None,
    guidance_scale: float = 1.0,
    prefix_steps: int | None = None,
    action_groups: dict[str, list[int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Return per-window metrics, averaging paired action errors over noise draws.

    Only the future-feature slot changes. Current VLM features, state, masks,
    flow weights, and each repeat's sampling noise stay identical across modes.
    """
    if backend.training:
        raise ValueError("Call backend.eval() before running offline diagnostics.")
    if not backend.model_cfg.future_prediction:
        raise ValueError("This diagnostic requires future_prediction=true.")
    if str(getattr(backend.model_cfg, "flow_future_feature", "predicted")) != "predicted":
        raise ValueError("Use a checkpoint trained with predicted future features, not current_feature.")
    if noise_repeats < 1:
        raise ValueError("noise_repeats must be positive.")
    if num_inference_steps is not None and num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive.")
    video = batch["primary_video"]
    if video.ndim != 5 or video.shape[1] < 2:
        raise ValueError("The diagnostic needs current and real future images [B,T,C,H,W], T>=2.")
    if prefix_steps is not None and not 1 <= prefix_steps <= batch["actions"].shape[1]:
        raise ValueError("prefix_steps must be between 1 and the action horizon.")

    prepared = backend._prepare_train_batch(batch=batch)
    device = prepared["input_ids"].device
    shared = backend._run_shared_encoding_train(
        prepared_batch=prepared, source="offline_teacher_diagnostic", lam_features_with_no_grad=True,
    )
    with _autocast(device, torch.bfloat16):
        teacher = backend._run_lam_teacher(
            primary_video=prepared["primary_video"], embodiment_id=prepared["embodiment_id"],
        )
        teacher_future = backend._decode_future_tokens_strict_single_query(
            h_t=shared.h_t, pred_action_emb=teacher, source="offline_teacher_diagnostic",
        )

    predicted = shared.pred_action_emb.float()
    teacher = teacher.float()
    metrics = {
        "latent_mse": _mse_per_sample(predicted, teacher),
        "latent_cosine": F.cosine_similarity(predicted, teacher, dim=-1).mean(dim=1),
        "teacher_latent_energy": teacher.square().flatten(1).mean(dim=1),
        "predicted_latent_energy": predicted.square().flatten(1).mean(dim=1),
    }
    futures = {
        "predicted": shared.h_t1_pred,
        "teacher_latent": teacher_future,
        "gt_future": shared.h_t1_gt,
    }
    for name, future in {**futures, "current": shared.h_t}.items():
        metrics[f"future_mse/{name}"] = _mse_per_sample(future, shared.h_t1_gt)

    target = prepared["actions"]
    action_mask = prepared["actions_mask"].bool()
    groups = {"all": list(range(target.shape[-1])), **(action_groups or {})}
    for name, indices in groups.items():
        if not indices or min(indices) < 0 or max(indices) >= target.shape[-1]:
            raise ValueError(f"Invalid action group {name!r}: {indices}")

    cuda_devices = [] if device.type != "cuda" else [device.index if device.index is not None else torch.cuda.current_device()]
    for repeat in range(noise_repeats):
        reference_actions = None
        for name in CONDITIONS:
            # Restore the caller's RNG and replay the same initial flow noise.
            with torch.random.fork_rng(devices=cuda_devices):
                torch.random.default_generator.manual_seed(noise_seed + repeat)
                if cuda_devices:
                    torch.cuda.default_generators[cuda_devices[0]].manual_seed(noise_seed + repeat)
                with _autocast(device, backend.flow._compute_dtype()):
                    actions = backend.flow.sample_actions_cfg(
                        h_t=shared.h_t,
                        h_t1_star=futures[name],
                        h_vlm=shared.h_vlm,
                        state=prepared["state"],
                        state_mask=prepared["state_mask"],
                        action_hz=prepared["action_hz"],
                        embodiment_id=prepared["embodiment_id"],
                        cfg_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        attention_mask=prepared["attention_mask"] == 1,
                        contact_geometry=prepared.get("contact_geometry"),
                        object_pose_condition=prepared.get("object_pose_condition"),
                        interaction_tokens=shared.interaction_tokens,
                        return_padded=True,
                    )
            if reference_actions is None:
                reference_actions = actions
            values = {}
            for group, indices in groups.items():
                values[f"action_mse/{name}/{group}"] = masked_error_per_sample(
                    actions[..., indices], target[..., indices], action_mask[..., indices],
                )
            values[f"action_mae/{name}"] = masked_error_per_sample(actions, target, action_mask, absolute=True)
            values[f"action_change_mse/{name}"] = masked_error_per_sample(actions, reference_actions, action_mask)
            if prefix_steps is not None:
                values[f"prefix_action_mse/{name}"] = masked_error_per_sample(
                    actions[:, :prefix_steps], target[:, :prefix_steps], action_mask[:, :prefix_steps],
                )
            for key, value in values.items():
                metrics[key] = metrics.get(key, torch.zeros_like(value)) + value / noise_repeats

    for key, values in metrics.items():
        if not torch.isfinite(values).all():
            raise ValueError(f"Nonfinite diagnostic metric: {key}")
    return {key: values.detach().cpu() for key, values in metrics.items()}
