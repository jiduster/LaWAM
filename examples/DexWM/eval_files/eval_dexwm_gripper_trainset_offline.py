from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = Path(__file__).resolve().parent
STARVLA_ROOT = SCRIPT_DIR.parents[2]
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

from examples.DexWM.eval_files.dexwm_common import unnormalize_dexwm_actions
from starVLA.model.framework.base_framework import baseframework


POSE_KEYS = (
    "robot0_right_eef_T_right_base_pos",
    "robot0_right_eef_pos",
    "robot0_right_eef_T_world_pos",
)
QUAT_KEYS = (
    "robot0_right_eef_T_right_base_quat_xyzw",
    "robot0_right_eef_quat",
    "robot0_right_eef_T_world_quat_xyzw",
)
PRIMARY_KEYS = ("robot0_robotview_2_image", "robot0_robotview_image")
WRIST_KEY = "gripper0_right_right_eye_in_hand_image"
ACTION_KEYS = ("abs_right_arm_base_action", "abs_action", "actions")
STATE_DIM = 25
ACTION_DIM = 25


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_json_attr(attrs: Any, key: str) -> dict[str, Any] | None:
    if key not in attrs:
        return None
    raw = attrs[key]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _resolve_lang(demo_group: Any, fallback: str) -> str:
    meta = demo_group.attrs.get("ep_meta", None)
    if isinstance(meta, bytes):
        meta = meta.decode("utf-8")
    if meta is not None:
        try:
            parsed = json.loads(meta)
            if isinstance(parsed, dict):
                lang = parsed.get("free_form_lang") or parsed.get("lang")
                if lang:
                    return str(lang)
        except Exception:
            pass
    return str(fallback)


def _find_key(group: Any, keys: Sequence[str], *, what: str) -> str:
    for key in keys:
        if key in group:
            return key
    raise KeyError(f"Missing {what}; expected one of {list(keys)}.")


def _state25_from_obs_group(obs_group: Any, idx: int) -> np.ndarray:
    pos_key = _find_key(obs_group, POSE_KEYS, what="EEF position")
    quat_key = _find_key(obs_group, QUAT_KEYS, what="EEF quaternion")
    if "robot0_right_gripper_qpos" not in obs_group:
        raise KeyError("Missing `robot0_right_gripper_qpos`.")

    pos = np.asarray(obs_group[pos_key][idx], dtype=np.float32).reshape(-1)
    quat = np.asarray(obs_group[quat_key][idx], dtype=np.float32).reshape(-1)
    gripper = np.asarray(obs_group["robot0_right_gripper_qpos"][idx], dtype=np.float32).reshape(-1)
    rot6 = R.from_quat(quat).as_matrix()[:2, :].reshape(-1).astype(np.float32)
    state = np.concatenate([pos, rot6, gripper], axis=0).astype(np.float32, copy=False)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"Expected {STATE_DIM}D state, got {state.shape}.")
    return state


def _raw_action25_from_demo(demo_group: Any, indices: Iterable[int]) -> np.ndarray:
    action_key = _find_key(demo_group, ACTION_KEYS, what="action")
    raw = np.asarray(demo_group[action_key], dtype=np.float32)
    rows = raw[[int(i) for i in indices]]
    if rows.ndim != 2 or rows.shape[-1] < 31:
        raise ValueError(f"Unexpected raw action shape {rows.shape}; expected [T, >=31].")
    pos = rows[:, :3]
    rot6 = R.from_rotvec(rows[:, 3:6]).as_matrix()[:, :2, :].reshape(rows.shape[0], 6)
    gripper = rows[:, 15:31]
    action = np.concatenate([pos, rot6.astype(np.float32), gripper], axis=-1).astype(np.float32, copy=False)
    if action.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM}D action, got {action.shape}.")
    return action


def _normalize_array(array: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    high_key = "max" if "max" in stats else "q99"
    low_key = "min" if "min" in stats else "q01"
    high = np.asarray(stats[high_key], dtype=np.float32)
    low = np.asarray(stats[low_key], dtype=np.float32)
    denom = high - low
    out = arr.copy()
    valid = np.abs(denom) > 1e-12
    out[..., valid] = (out[..., valid] - low[valid]) / denom[valid] * 2.0 - 1.0
    out[..., ~valid] = 0.0
    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


def _select_start_indices(length: int, horizon: int, max_steps: int, stride: int) -> list[int]:
    last_start = max(0, int(length) - 1)
    candidates = list(range(0, last_start + 1, max(1, int(stride))))
    if not candidates:
        candidates = [0]
    if max_steps > 0:
        candidates = candidates[: int(max_steps)]
    return [int(i) for i in candidates if i < length]


def _select_video_indices(length: int, stride: int, max_frames: int) -> list[int]:
    length = max(1, int(length))
    stride = max(1, int(stride))
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    if max_frames > 0 and len(indices) > int(max_frames):
        return np.rint(np.linspace(0, length - 1, num=int(max_frames))).astype(np.int64).tolist()
    return [int(i) for i in indices]


def _episode_sort_key(name: str) -> tuple[int, int | str]:
    prefix, sep, suffix = str(name).rpartition("_")
    if sep and suffix.isdigit():
        return (0, int(suffix))
    return (1, str(name))


def _build_example(
    *,
    obs_group: Any,
    lang: str,
    start_idx: int,
    state_stats: dict[str, Any],
    embodiment_id: int,
    action_hz: float,
    wm_primary_video_key: str | None,
    normalize_state: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    primary_images = []
    for key in PRIMARY_KEYS:
        if key in obs_group:
            primary_images.append(np.asarray(obs_group[key][start_idx], dtype=np.uint8))
    if not primary_images:
        raise KeyError(f"Missing primary image keys: {list(PRIMARY_KEYS)}.")

    raw_state = _state25_from_obs_group(obs_group, start_idx)
    model_state = _normalize_array(raw_state, state_stats) if normalize_state else raw_state
    example: dict[str, Any] = {
        "lang": str(lang),
        "primary_image": primary_images,
        "state": model_state,
        "embodiment_id": int(embodiment_id),
        "action_hz": float(action_hz),
    }
    if wm_primary_video_key:
        if wm_primary_video_key not in obs_group:
            raise KeyError(f"Missing WM primary image key: {wm_primary_video_key}.")
        example["wm_primary_image"] = np.asarray(obs_group[wm_primary_video_key][start_idx], dtype=np.uint8)
    if WRIST_KEY in obs_group:
        example["wrist_image"] = [np.asarray(obs_group[WRIST_KEY][start_idx], dtype=np.uint8)]
    return example, raw_state


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _mean_abs_cos(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    diff = pred - gt
    pred_flat = pred.reshape(pred.shape[0], -1) if pred.ndim > 1 else pred.reshape(1, -1)
    gt_flat = gt.reshape(gt.shape[0], -1) if gt.ndim > 1 else gt.reshape(1, -1)
    denom = np.linalg.norm(pred_flat, axis=-1) * np.linalg.norm(gt_flat, axis=-1)
    valid = denom > 1e-8
    cosine = np.full(pred_flat.shape[0], np.nan, dtype=np.float32)
    cosine[valid] = np.sum(pred_flat[valid] * gt_flat[valid], axis=-1) / denom[valid]
    return {
        "mse": float(np.mean(diff**2)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(math.sqrt(float(np.mean(diff**2)))),
        "cosine": float(np.nanmean(cosine)) if np.any(valid) else float("nan"),
        "pred_l2": float(np.mean(np.linalg.norm(pred_flat, axis=-1))),
        "gt_l2": float(np.mean(np.linalg.norm(gt_flat, axis=-1))),
    }


def _component_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        "pos": _mean_abs_cos(pred[..., 0:3], gt[..., 0:3]),
        "rot6d": _mean_abs_cos(pred[..., 3:9], gt[..., 3:9]),
        "gripper": _mean_abs_cos(pred[..., 9:25], gt[..., 9:25]),
    }


def _write_demo_artifacts(
    *,
    output_dir: Path,
    demo_name: str,
    obs_group: Any,
    start_indices: Sequence[int],
    video_indices: Sequence[int] | None,
    video_fps: float,
    pred_first: np.ndarray,
    gt_first: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    import cv2
    import imageio.v2 as imageio

    demo_dir = output_dir / demo_name
    demo_dir.mkdir(parents=True, exist_ok=True)
    (demo_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
    np.save(demo_dir / "pred_first_actions.npy", np.asarray(pred_first, dtype=np.float32))
    np.save(demo_dir / "gt_first_actions.npy", np.asarray(gt_first, dtype=np.float32))

    def write_video_or_frames(name: str, frames: list[np.ndarray]) -> None:
        if not frames:
            return
        try:
            out_path = demo_dir / f"{name}.mp4"
            first = np.asarray(frames[0], dtype=np.uint8)
            if first.ndim != 3 or first.shape[2] != 3:
                raise ValueError(f"Expected RGB frames with shape [H, W, 3], got {first.shape}.")
            height, width = int(first.shape[0]), int(first.shape[1])
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(video_fps),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"cv2.VideoWriter could not open `{out_path}`.")
            try:
                for frame in frames:
                    frame_u8 = np.asarray(frame, dtype=np.uint8)
                    if frame_u8.shape[:2] != (height, width):
                        raise ValueError(
                            f"All frames must share the same shape; expected {(height, width)}, got {frame_u8.shape[:2]}."
                        )
                    writer.write(cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            return
        except Exception as exc:
            (demo_dir / f"{name}_video_error.txt").write_text(str(exc), encoding="utf-8")
        for idx, frame in enumerate(frames[:8]):
            imageio.imwrite(demo_dir / f"{name}_{idx:03d}.png", frame)

    frame_indices = list(video_indices) if video_indices is not None else list(start_indices)

    if "robot0_robotview_2_image" in obs_group:
        frames = [np.asarray(obs_group["robot0_robotview_2_image"][i], dtype=np.uint8) for i in frame_indices]
        write_video_or_frames("robotview_sampled", frames)
    if WRIST_KEY in obs_group:
        frames = [np.asarray(obs_group[WRIST_KEY][i], dtype=np.uint8) for i in frame_indices]
        write_video_or_frames("wrist_sampled", frames)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        x = np.asarray(start_indices)
        groups = [("pos", slice(0, 3)), ("rot6d", slice(3, 9)), ("gripper", slice(9, 25))]
        for ax, (name, slc) in zip(axes, groups):
            pred_mean = pred_first[:, slc].mean(axis=1)
            gt_mean = gt_first[:, slc].mean(axis=1)
            ax.plot(x, gt_mean, label=f"gt {name}", linewidth=2)
            ax.plot(x, pred_mean, label=f"pred {name}", linewidth=2, linestyle="--")
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
        axes[-1].set_xlabel("dataset timestep")
        fig.tight_layout()
        fig.savefig(demo_dir / "action_compare.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        (demo_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/mnt/ceph3/dexwm/robocasa_random_data/gripper_open_and_close/combine_demos_0.hdf5"),
    )
    parser.add_argument("--datasets", nargs="*", type=Path, default=None)
    parser.add_argument("--episodes", nargs="*", default=["demo_0", "demo_1", "demo_10"])
    parser.add_argument("--max_steps_per_episode", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/dexwm_trainset_offline/gripper_open_and_close"))
    parser.add_argument(
        "--save_full_episode_video",
        action="store_true",
        help="Save visualization videos from the full source episode instead of only eval-sampled frames.",
    )
    parser.add_argument("--video_stride", type=int, default=2)
    parser.add_argument("--video_max_frames", type=int, default=240)
    parser.add_argument("--video_fps", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--unnorm_key", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no_normalize_state", action="store_true")
    return parser


def _resolve_unnorm_key(norm_stats: dict[str, Any], unnorm_key: str | None) -> str:
    if unnorm_key is None:
        if len(norm_stats) != 1:
            raise ValueError(f"Please pass --unnorm_key from {list(norm_stats.keys())}.")
        return str(next(iter(norm_stats)))
    if unnorm_key not in norm_stats:
        raise ValueError(f"Unknown --unnorm_key {unnorm_key!r}; available keys: {list(norm_stats.keys())}.")
    return str(unnorm_key)


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    policy = baseframework.from_pretrained(str(args.ckpt_path))
    if args.use_bf16:
        policy = policy.to(torch.bfloat16)
    policy = policy.to(args.device).eval()

    norm_stats = getattr(policy, "norm_stats", None) or {}
    unnorm_key = _resolve_unnorm_key(norm_stats, args.unnorm_key)
    state_stats = norm_stats[unnorm_key]["state"]
    action_stats = norm_stats[unnorm_key]["action"]
    action_hz = float(getattr(policy.config.datasets.vla_data, "action_hz_override", 1.0) or 1.0)
    wm_primary_video_key = str(
        getattr(policy.config.datasets.vla_data, "wm_primary_video_key", "robot0_robotview_2_image")
    )
    embodiment_id = 31
    if hasattr(policy.config.datasets.vla_data, "embodiment_id"):
        embodiment_id = int(policy.config.datasets.vla_data.embodiment_id)

    dataset_paths = [Path(p) for p in args.datasets] if args.datasets else [Path(args.dataset)]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_first_pred: list[np.ndarray] = []
    all_first_gt: list[np.ndarray] = []
    all_chunk_pred: list[np.ndarray] = []
    all_chunk_gt: list[np.ndarray] = []
    demo_summaries: dict[str, Any] = {}

    use_dataset_subdirs = len(dataset_paths) > 1
    for dataset_path in dataset_paths:
        artifact_root = output_dir / dataset_path.stem if use_dataset_subdirs else output_dir
        with h5py.File(dataset_path, "r") as h5:
            env_args = _load_json_attr(h5["data"].attrs, "env_args")
            available = set(h5["data"].keys())
            requested_episodes = (
                sorted(available, key=_episode_sort_key)
                if len(args.episodes) == 1 and str(args.episodes[0]).lower() == "all"
                else list(args.episodes)
            )
            matched_any = False
            for demo_name in requested_episodes:
                if demo_name not in available:
                    if use_dataset_subdirs:
                        continue
                    raise KeyError(f"Episode {demo_name!r} not found in {dataset_path}.")
                matched_any = True
                demo = h5["data"][demo_name]
                obs_group = demo["obs"]
                length = int(demo["actions"].shape[0] if "actions" in demo else demo.attrs.get("num_samples", 0))
                if length <= 0:
                    raise ValueError(f"Episode {demo_name} has invalid length={length}.")
                horizon = int(getattr(policy.config.datasets.vla_data, "action_horizon", 4))
                starts = _select_start_indices(
                    length=length,
                    horizon=horizon,
                    max_steps=int(args.max_steps_per_episode),
                    stride=int(args.stride),
                )
                lang = _resolve_lang(demo, fallback="dexwm robocasa")

                examples = []
                raw_states = []
                gt_chunks = []
                for start_idx in starts:
                    example, raw_state = _build_example(
                        obs_group=obs_group,
                        lang=lang,
                        start_idx=start_idx,
                        state_stats=state_stats,
                        embodiment_id=embodiment_id,
                        action_hz=action_hz,
                        wm_primary_video_key=wm_primary_video_key,
                        normalize_state=not bool(args.no_normalize_state),
                    )
                    action_indices = [min(start_idx + off, length - 1) for off in range(horizon)]
                    gt_chunk = _raw_action25_from_demo(demo, action_indices)
                    examples.append(example)
                    raw_states.append(raw_state)
                    gt_chunks.append(gt_chunk)

                pred_chunks = []
                with torch.inference_mode():
                    for batch_examples in _batched(examples, int(args.batch_size)):
                        out = policy.predict_action(
                            examples=batch_examples,
                            guidance_scale=args.guidance_scale,
                            num_inference_steps=args.num_inference_steps,
                        )
                        normalized = np.asarray(out["normalized_actions"], dtype=np.float32)
                        raw_pred = unnormalize_dexwm_actions(normalized, action_stats)
                        pred_chunks.extend([raw_pred[i] for i in range(raw_pred.shape[0])])

                pred_arr = np.asarray(pred_chunks, dtype=np.float32)
                gt_arr = np.asarray(gt_chunks, dtype=np.float32)
                pred_first = pred_arr[:, 0, :]
                gt_first = gt_arr[:, 0, :]
                demo_metrics = {
                    "dataset": str(dataset_path),
                    "episode": demo_name,
                    "env_name": None if env_args is None else env_args.get("env_name"),
                    "language": lang,
                    "num_samples": int(len(starts)),
                    "start_indices": starts,
                    "video_indices": (
                        _select_video_indices(length, int(args.video_stride), int(args.video_max_frames))
                        if bool(args.save_full_episode_video)
                        else starts
                    ),
                    "state_normalized": not bool(args.no_normalize_state),
                    "first_action": {
                        "overall": _mean_abs_cos(pred_first, gt_first),
                        "components": _component_metrics(pred_first, gt_first),
                    },
                    "chunk": {
                        "overall": _mean_abs_cos(pred_arr.reshape(-1, ACTION_DIM), gt_arr.reshape(-1, ACTION_DIM)),
                        "components": _component_metrics(
                            pred_arr.reshape(-1, ACTION_DIM),
                            gt_arr.reshape(-1, ACTION_DIM),
                        ),
                    },
                    "raw_state_mean_l2": float(np.mean(np.linalg.norm(np.asarray(raw_states), axis=-1))),
                }
                _write_demo_artifacts(
                    output_dir=artifact_root,
                    demo_name=demo_name,
                    obs_group=obs_group,
                    start_indices=starts,
                    video_indices=demo_metrics["video_indices"],
                    video_fps=float(args.video_fps),
                    pred_first=pred_first,
                    gt_first=gt_first,
                    metrics=demo_metrics,
                )
                summary_key = f"{dataset_path.stem}/{demo_name}" if use_dataset_subdirs else demo_name
                demo_summaries[summary_key] = demo_metrics
                all_first_pred.append(pred_first)
                all_first_gt.append(gt_first)
                all_chunk_pred.append(pred_arr.reshape(-1, ACTION_DIM))
                all_chunk_gt.append(gt_arr.reshape(-1, ACTION_DIM))
            if not matched_any:
                raise KeyError(
                    f"None of the requested episodes were found in {dataset_path}. "
                    f"First available episodes: {sorted(available, key=_episode_sort_key)[:10]}"
                )

    if not all_first_pred:
        raise RuntimeError("No episodes were evaluated.")
    first_pred = np.concatenate(all_first_pred, axis=0)
    first_gt = np.concatenate(all_first_gt, axis=0)
    chunk_pred = np.concatenate(all_chunk_pred, axis=0)
    chunk_gt = np.concatenate(all_chunk_gt, axis=0)
    summary = {
        "ckpt_path": str(args.ckpt_path),
        "dataset": str(dataset_paths[0]) if len(dataset_paths) == 1 else [str(path) for path in dataset_paths],
        "output_dir": str(output_dir),
        "unnorm_key": unnorm_key,
        "state_normalized": not bool(args.no_normalize_state),
        "episodes": demo_summaries,
        "aggregate": {
            "first_action": {
                "overall": _mean_abs_cos(first_pred, first_gt),
                "components": _component_metrics(first_pred, first_gt),
            },
            "chunk": {
                "overall": _mean_abs_cos(chunk_pred, chunk_gt),
                "components": _component_metrics(chunk_pred, chunk_gt),
            },
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary["aggregate"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
