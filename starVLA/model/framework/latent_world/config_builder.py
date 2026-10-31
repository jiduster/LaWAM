from __future__ import annotations

from dataclasses import fields
import math
from typing import Any

from starVLA.model.framework.vlas.flowmatching_expert import ConditionalFlowMatchingConfig
from starVLA.model.framework.vlas.lawam import LatentWorldPolicyConfig


class LatentWorldPolicyConfigBuilder:
    """Build LatentWorldPolicyConfig from global training config with strict field validation."""

    _DEPRECATED_FIELDS = {"model_id", "yaml_path"}
    _DATASET_FIELDS = {"enable_primary_video_aug", "enable_primary_random_resized_crop"}
    _WINDOW_KEYS = {"future_action_window_size", "past_action_window_size", "action_horizon"}
    _MIGRATED_FREEZE_FIELDS = {
        "freeze_vision_backbone",
        "freeze_llm_backbone",
        "freeze_last_llm_layer",
        "freeze_embedding",
        "unfreeze_vision_merger",
        "keep_llm_first_n_layers",
        "unfreeze_llm_last_n_layers",
        "unfreeze_lam_decoder",
    }

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def build(self) -> LatentWorldPolicyConfig:
        policy_cfg = LatentWorldPolicyConfig()
        framework_cfg = self.cfg.framework
        if "latent_world" in framework_cfg:
            raise ValueError(
                "`framework.latent_world` is no longer supported. "
                "Use `framework.action_model` with required keys "
                "`future_action_window_size`, `past_action_window_size`, and `action_horizon`."
            )
        if "action_model" not in framework_cfg:
            raise ValueError(
                "Missing `framework.action_model` for LaWAM. "
                "Please configure LatentWorld parameters under `framework.action_model`."
            )
        am = framework_cfg.action_model

        missing_window_keys = sorted(k for k in self._WINDOW_KEYS if am.get(k, None) is None)
        if missing_window_keys:
            raise ValueError(
                "Missing required action chunk fields in `framework.action_model`: "
                f"{missing_window_keys}."
            )
        future_window = int(am.future_action_window_size)
        past_window = int(am.past_action_window_size)
        action_horizon = int(am.action_horizon)
        expected_action_horizon = int(future_window + past_window + 1)
        if action_horizon != expected_action_horizon:
            raise ValueError(
                "Invalid action chunk config: expected "
                "`action_horizon = future_action_window_size + past_action_window_size + 1`, got "
                f"action_horizon={action_horizon}, future_action_window_size={future_window}, "
                f"past_action_window_size={past_window}."
            )

        top_keys = set(am.keys())
        deprecated = sorted(self._DEPRECATED_FIELDS.intersection(top_keys))
        if deprecated:
            raise ValueError(
                "Deprecated action_model fields are not allowed for LaWAM: "
                f"{deprecated}. Use `framework.qwenvl.base_vlm` as VLM source."
            )

        migrated_freeze = sorted(self._MIGRATED_FREEZE_FIELDS.intersection(top_keys))
        if migrated_freeze:
            raise ValueError(
                "LatentWorld freeze fields in `framework.action_model` are not allowed. "
                "Use `trainer.freeze`: "
                f"{migrated_freeze}."
            )

        misplaced_dataset_fields = sorted(self._DATASET_FIELDS.intersection(top_keys))
        if misplaced_dataset_fields:
            raise ValueError(
                "Dataset-side LatentWorld fields are not allowed in `framework.action_model`. "
                "Move them to `datasets.vla_data`: "
                f"{misplaced_dataset_fields}."
            )

        valid_top_keys = {f.name for f in fields(LatentWorldPolicyConfig)} - {"flow_cfg"}
        unknown_top = sorted(k for k in top_keys if k not in valid_top_keys and k not in self._WINDOW_KEYS and k != "flow_cfg")
        if unknown_top:
            raise ValueError(f"Unknown `framework.action_model` fields for LatentWorld: {unknown_top}")

        for key in top_keys:
            if key != "flow_cfg" and key not in self._WINDOW_KEYS:
                setattr(policy_cfg, key, am[key])

        policy_cfg.future_action_window_size = future_window
        policy_cfg.past_action_window_size = past_window
        policy_cfg.action_horizon = action_horizon

        flow_cfg = am.get("flow_cfg", None)
        if "flow_cfg" in top_keys and flow_cfg is not None:
            flow_keys = set(flow_cfg.keys())
            if "window_size" in flow_keys:
                raise ValueError(
                    "`framework.action_model.flow_cfg.window_size` has been removed. "
                    "Configure only `framework.action_model.action_horizon`."
                )
            legacy_flow_keys = sorted(k for k in ("proprio_dim", "use_proprio") if k in flow_keys)
            if legacy_flow_keys:
                raise ValueError(
                    "Legacy flow_cfg fields are not supported: "
                    f"{legacy_flow_keys}. Use `state_dim` / `use_state`."
                )
            deprecated_flow_keys = sorted(k for k in ("state_dropout_prob",) if k in flow_keys)
            if deprecated_flow_keys:
                raise ValueError(
                    "Deprecated flow_cfg fields are not supported: "
                    f"{deprecated_flow_keys}. Use `use_state` to enable or disable state conditioning."
                )
            valid_flow_keys = {f.name for f in fields(ConditionalFlowMatchingConfig)}
            unknown_flow = sorted(k for k in flow_keys if k not in valid_flow_keys)
            if unknown_flow:
                raise ValueError(f"Unknown `framework.action_model.flow_cfg` fields: {unknown_flow}")
            for key in flow_keys:
                setattr(policy_cfg.flow_cfg, key, flow_cfg[key])

        # Validate head selection before any model or dataloader is constructed.
        _ = policy_cfg.num_vlm_action_queries
        from starVLA.interaction_geometry import ENTITY_NAMES, FEATURE_DIM
        from starVLA.model.framework.vlas.interaction_tokens import AUX_MODES, MODES, PRED_TOKEN_MODES
        from starVLA.model.framework.vlas.wm_geometry import FUTURE_GEOMETRY_MODES, WM_MODES
        mode = str(policy_cfg.interaction_mode)
        future_mode = mode in FUTURE_GEOMETRY_MODES
        labels = mode in {"oracle", *AUX_MODES, *WM_MODES}
        if mode not in {*MODES, *WM_MODES}:
            raise ValueError("Unknown interaction_mode")
        flow = policy_cfg.flow_cfg
        if bool(flow.interaction_conditioning) != (mode in MODES[1:]):
            raise ValueError("interaction_conditioning is enabled only for oracle/visual/visual_aux/visual_pred_aux/visual_joint_aux")
        if (future_mode or mode in PRED_TOKEN_MODES) and (not policy_cfg.future_prediction or policy_cfg.flow_future_feature != 'predicted'
                        or policy_cfg.enable_flow_h_t1_scheduled_sampling):
            raise ValueError('Future geometry/predicted visual tokens require predicted future features without scheduled sampling')
        if mode != "none":
            if policy_cfg.action_expert_only:
                raise ValueError("Interaction experiments require full Stage 2: action_expert_only=false")
            task = str(policy_cfg.interaction_task)
            if task not in ENTITY_NAMES or int(policy_cfg.interaction_entities) != len(ENTITY_NAMES[task]):
                raise ValueError("Interaction task/entity schema mismatch")
            if int(policy_cfg.interaction_feature_dim) != FEATURE_DIM:
                raise ValueError("Interaction feature dimension does not match schema")
            dim, heads = int(flow.interaction_token_dim), int(flow.interaction_heads)
            if min(dim, heads, int(policy_cfg.interaction_queries), int(policy_cfg.interaction_layers)) < 1 or dim % heads:
                raise ValueError("Invalid interaction token dimensions/heads/layers")
            if not 1 <= int(flow.interaction_every_n_blocks) <= int(flow.num_layers):
                raise ValueError("Invalid interaction attention interval")
            if labels and not policy_cfg.interaction_sidecar_root:
                raise ValueError("Geometry supervision requires interaction_sidecar_root")
            if labels and (policy_cfg.interaction_feature_mean is None or policy_cfg.interaction_feature_std is None):
                raise ValueError("Geometry experiments require training-only feature mean/std")
        weight = float(policy_cfg.interaction_loss_weight)
        auxiliary = mode in {*AUX_MODES, *WM_MODES}
        if not math.isfinite(weight) or (auxiliary and weight <= 0) or (not auxiliary and weight != 0):
            raise ValueError("Positive interaction_loss_weight is required only for geometry auxiliary modes")
        if policy_cfg.plan_pose_conditioning:
            if policy_cfg.action_expert_only:
                raise ValueError("Planning pose requires action_expert_only=false")
            if not policy_cfg.future_prediction or str(policy_cfg.flow_future_feature).lower() != "predicted":
                raise ValueError("Planning pose requires future_prediction=true and flow_future_feature=predicted")
            if policy_cfg.plan_pose_position_mean is None or policy_cfg.plan_pose_position_std is None:
                raise ValueError("Planning pose requires training-only plan_pose_position_mean/std; use train_pose_plan.sh")
            dim, heads = int(policy_cfg.plan_pose_token_dim), int(policy_cfg.plan_pose_attention_heads)
            if dim <= 0 or heads <= 0 or dim % heads:
                raise ValueError("plan_pose_token_dim must be positive and divisible by plan_pose_attention_heads")
        return policy_cfg
