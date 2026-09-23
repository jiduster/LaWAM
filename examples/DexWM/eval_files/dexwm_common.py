from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R


def cfg_get(data_cfg: Any, key: str, default: Any = None) -> Any:
    if data_cfg is None:
        return default
    if hasattr(data_cfg, "get"):
        return data_cfg.get(key, default)
    return getattr(data_cfg, key, default)


def normalize_quaternion_value(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float32)
    if quat.ndim == 1:
        quat = quat[None, :]

    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    normalized = np.zeros_like(quat, dtype=np.float32)
    valid_rows = np.isfinite(norm[:, 0]) & (norm[:, 0] > 1e-8)
    if np.any(valid_rows):
        normalized[valid_rows] = quat[valid_rows] / norm[valid_rows]
    if np.any(~valid_rows):
        normalized[~valid_rows, 0] = 1.0
    return normalized


@dataclass(frozen=True)
class DexWMControlSpec:
    data_mix: str
    action_hz: float
    env_action_dim: int
    right_arm_slice: tuple[int, int]
    right_gripper_slice: tuple[int, int]


def resolve_dexwm_control_from_data_mix(data_mix: Any) -> DexWMControlSpec:
    normalized = str(data_mix).strip().lower()
    if not normalized.startswith("dexwm_robocasa"):
        raise ValueError(f"Unsupported DexWM data_mix for eval: {normalized!r}")
    return DexWMControlSpec(
        data_mix=normalized,
        action_hz=1.0,
        env_action_dim=47,
        right_arm_slice=(0, 6),
        right_gripper_slice=(15, 31),
    )


def extract_dexwm_data_mix(model_config: dict[str, Any]) -> str:
    datasets_cfg = model_config.get("datasets", {})
    if not isinstance(datasets_cfg, dict):
        raise ValueError("Checkpoint config is missing `datasets` dict.")
    vla_data_cfg = datasets_cfg.get("vla_data", {})
    if not isinstance(vla_data_cfg, dict):
        raise ValueError("Checkpoint config is missing `datasets.vla_data`.")
    data_mix = vla_data_cfg.get("data_mix", None)
    if data_mix is None:
        raise ValueError("Checkpoint config is missing `datasets.vla_data.data_mix`.")
    return str(data_mix)


def rot_mat_six_dim_to_axisangle(rot_mat_six_dim: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot_mat_six_dim, dtype=np.float32).reshape(-1)
    if rot.shape[0] != 6:
        raise ValueError(f"Expected 6D rotation vector, got shape {rot.shape}.")
    row1 = rot[0:3].astype(np.float64)
    row2 = rot[3:6].astype(np.float64)
    row1_norm = np.linalg.norm(row1)
    if not np.isfinite(row1_norm) or row1_norm < 1e-8:
        row1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        row1 = row1 / row1_norm
    row2 = row2 - np.dot(row1, row2) * row1
    row2_norm = np.linalg.norm(row2)
    if not np.isfinite(row2_norm) or row2_norm < 1e-8:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(row1, fallback))) > 0.95:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        row2 = fallback - np.dot(row1, fallback) * row1
        row2 = row2 / max(np.linalg.norm(row2), 1e-8)
    else:
        row2 = row2 / row2_norm
    row3 = np.cross(row1, row2)
    rotation_matrix = np.stack([row1, row2, row3], axis=0).astype(np.float32)
    return R.from_matrix(rotation_matrix).as_rotvec().astype(np.float32)


def flatten_dexwm_state(observation: dict[str, Any]) -> np.ndarray:
    pos = observation.get("robot0_right_eef_T_right_base_pos", observation.get("robot0_right_eef_pos", None))
    quat = observation.get("robot0_right_eef_T_right_base_quat_xyzw", observation.get("robot0_right_eef_quat", None))
    gripper = observation.get("robot0_right_gripper_qpos", None)
    if pos is None or quat is None or gripper is None:
        raise KeyError("DexWM eval observation is missing required proprioception keys.")
    pos = np.asarray(pos, dtype=np.float32).reshape(-1)
    quat = np.asarray(quat, dtype=np.float32).reshape(-1)
    gripper = np.asarray(gripper, dtype=np.float32).reshape(-1)
    rot6 = R.from_quat(quat).as_matrix()[:2, :].reshape(-1).astype(np.float32)
    state = np.concatenate([pos, rot6, gripper], axis=0)
    if state.shape[0] != 25:
        raise ValueError(f"DexWM flattened state must be 25D, got {state.shape[0]}.")
    return state.astype(np.float32, copy=False)


def build_dexwm_example(
    task_description: str,
    observation: dict[str, Any],
    *,
    wm_primary_video_key: str | None = None,
) -> dict[str, Any]:
    obs = observation.get("observation", observation)
    primary = obs.get("robot0_robotview_2_image", None)
    secondary = obs.get("robot0_robotview_image", None)
    wrist = obs.get("gripper0_right_right_eye_in_hand_image", None)
    if primary is None:
        raise KeyError("DexWM eval example requires `robot0_robotview_2_image`.")
    images = [np.asarray(primary)]
    if secondary is not None:
        images.append(np.asarray(secondary))
    wrist_images = [np.asarray(wrist)] if wrist is not None else []
    example = {
        "lang": str(task_description),
        "primary_image": images,
        "state": flatten_dexwm_state(obs),
    }
    if wm_primary_video_key:
        wm_primary = obs.get(str(wm_primary_video_key), None)
        if wm_primary is None:
            raise KeyError(f"DexWM eval example requires WM primary view `{wm_primary_video_key}`.")
        example["wm_primary_image"] = np.asarray(wm_primary)
    if wrist_images:
        example["wrist_image"] = wrist_images
    return example


def remap_dexwm_action_to_env(action: np.ndarray, control_spec: DexWMControlSpec) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] < 25:
        raise ValueError(f"DexWM model action must be at least 25D, got {action.shape[0]}.")
    env_action = np.zeros(control_spec.env_action_dim, dtype=np.float32)
    rot_axisangle = rot_mat_six_dim_to_axisangle(action[3:9])
    env_action[control_spec.right_arm_slice[0] : control_spec.right_arm_slice[1]] = np.concatenate(
        [action[0:3], rot_axisangle],
        axis=0,
    )
    env_action[
        control_spec.right_gripper_slice[0] : control_spec.right_gripper_slice[1]
    ] = action[9:25]
    return env_action


def unnormalize_dexwm_actions(normalized_actions: np.ndarray, action_stats: dict[str, Any]) -> np.ndarray:
    normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
    if normalized_actions.ndim == 1:
        normalized_actions = normalized_actions[None, :]
    high_key = "max" if "max" in action_stats else "q99"
    low_key = "min" if "min" in action_stats else "q01"
    high = np.asarray(action_stats[high_key], dtype=np.float32)
    low = np.asarray(action_stats[low_key], dtype=np.float32)
    if normalized_actions.shape[-1] != int(high.shape[0]):
        if normalized_actions.shape[-1] < int(high.shape[0]):
            raise ValueError(
                "DexWM action dimension is smaller than action statistics dimension: "
                f"action_dim={normalized_actions.shape[-1]}, stats_dim={int(high.shape[0])}."
            )
        normalized_actions = normalized_actions[..., : int(high.shape[0])]
    mask = action_stats.get("mask", np.ones_like(low, dtype=bool))
    clipped = np.clip(normalized_actions, -1.0, 1.0)
    denorm = 0.5 * (clipped + 1.0) * (high - low) + low
    return np.where(mask, denorm, clipped).astype(np.float32, copy=False)
