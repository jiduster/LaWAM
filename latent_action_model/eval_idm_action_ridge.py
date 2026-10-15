"""Frozen-IDM ridge alignment on DexJoCo bimanual_photograph.

Both IDMs stay frozen. A closed-form ridge readout is fit on one split of
photograph episodes and scored on the remaining episodes. The action target is
the normalized 1.6 s chunk that takes the ego frame at t to the ego frame at
t+T, the same interval the IDM was trained to encode.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_action_model.core.lam_model import load_latent_action_model
from latent_action_model.data_loader.lerobot_dataset import _build_modality_config
from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.data_config_lam import build_lam_dataset_transform
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform


RELEASE_CKPT = (
    ROOT
    / "latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt"
)
RELEASE_YAML = ROOT / "latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml"
C_CKPT = (
    ROOT
    / "latent_action_model/logs/dino_large_vae_dexjoco_bimanual_exclude_photograph_fingertip_stage1/pytorch_model.pt"
)
DATA_ROOT = Path("/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets")
DATASET_NAME = "bimanual_photograph"
ROBOT_TYPE = "dexjoco_bimanual"
VIDEO_KEY = "video.primary_view"
STATE_KEY = "state.proprio"
ACTION_KEY = "action.proprio"
FRAME_DT_SEC = 1.6
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "latent_action_model/logs/idm_action_ridge")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--windows-per-episode", type=int, default=8)
    parser.add_argument("--fit-ratio", type=float, default=0.7)
    parser.add_argument("--frame-dt-sec", type=float, default=FRAME_DT_SEC)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def window_starts(length: int, stride: int, n_windows: int) -> list[int]:
    """Starts whose future frame t+stride and the intervening actions stay in-episode."""
    last = int(length) - int(stride) - 1
    if last < 0:
        return []
    count = max(1, int(n_windows))
    if last == 0 or count == 1:
        return [0]
    raw = np.linspace(0, last, count)
    return sorted({int(round(value)) for value in raw})


def build_action_transform(dataset: LeRobotSingleDataset) -> ComposedModalityTransform:
    base = ROBOT_TYPE_CONFIG_MAP[ROBOT_TYPE].transform(image_hw=(256, 256))
    action_keys = list(dataset.modality_keys["action"])
    kept = []
    for transform in base.transforms:
        apply_to = list(getattr(transform, "apply_to", []) or [])
        selected = [key for key in apply_to if key in action_keys]
        if not selected:
            continue
        copied = copy.deepcopy(transform)
        copied.apply_to = selected
        kept.append(copied)
    if not kept:
        raise RuntimeError("DexJoCo action normalization transforms were not found.")
    composed = ComposedModalityTransform(transforms=kept)
    composed.set_metadata(dataset.metadata)
    composed.eval()
    return composed


def build_dataset(args: argparse.Namespace, stride: int | None = None) -> tuple[LeRobotSingleDataset, int]:
    modality = _build_modality_config(
        robot_type=ROBOT_TYPE,
        num_frames=2,
        preferred_video_key=VIDEO_KEY,
        state_keys=[STATE_KEY],
        include_action=True,
    )
    dataset_name = getattr(args, "dataset_name", DATASET_NAME)
    dataset = LeRobotSingleDataset(
        dataset_path=args.data_root / dataset_name,
        modality_configs=modality,
        embodiment_tag=ROBOT_TYPE_TO_EMBODIMENT_TAG[ROBOT_TYPE],
        mode="all",
        video_backend=args.video_backend,
        data_cfg={"data_root_dir": str(args.data_root), "video_backend": args.video_backend},
    )
    fps = float(dataset.action_hz)
    resolved_stride = max(1, int(round(float(args.frame_dt_sec) * fps)))
    if stride is not None and resolved_stride != stride:
        raise RuntimeError(f"Stride changed from {stride} to {resolved_stride}.")
    dataset._delta_indices[VIDEO_KEY] = np.asarray([0, resolved_stride], dtype=np.int64)
    dataset._delta_indices[STATE_KEY] = np.asarray([0, resolved_stride], dtype=np.int64)
    dataset._delta_indices[ACTION_KEY] = np.arange(resolved_stride, dtype=np.int64)
    return dataset, resolved_stride


def split_episodes(episode_ids: list[int], fit_ratio: float) -> tuple[list[int], list[int]]:
    ordered = sorted(int(episode_id) for episode_id in episode_ids)
    if len(ordered) < 2:
        raise RuntimeError(f"Need at least 2 photograph episodes, found {len(ordered)}.")
    n_fit = int(round(len(ordered) * float(fit_ratio)))
    n_fit = min(max(n_fit, 1), len(ordered) - 1)
    return ordered[:n_fit], ordered[n_fit:]


def collect_windows(args: argparse.Namespace, dataset: LeRobotSingleDataset, stride: int) -> dict:
    lam_transform = build_lam_dataset_transform(
        robot_type=ROBOT_TYPE,
        image_hw=(256, 256),
        state_keys=[STATE_KEY],
        video_keys=[VIDEO_KEY],
    )
    lam_transform.set_metadata(dataset.metadata)
    lam_transform.eval()
    action_transform = build_action_transform(dataset)

    frames = []
    proprio = []
    actions = []
    episode_ids = []
    base_indices = []
    usable = []
    for episode_id, length in zip(dataset.trajectory_ids, dataset.trajectory_lengths):
        episode_id = int(episode_id)
        starts = window_starts(int(length), stride, args.windows_per_episode)
        if not starts:
            continue
        usable.append(episode_id)
        for start in starts:
            raw = dataset.get_step_data(episode_id, start)
            obs = lam_transform({VIDEO_KEY: raw[VIDEO_KEY], STATE_KEY: raw[STATE_KEY]})
            act = action_transform({ACTION_KEY: raw[ACTION_KEY]})
            frame = as_numpy(obs[VIDEO_KEY])
            state = as_numpy(obs[STATE_KEY])
            action = as_numpy(act[ACTION_KEY])
            if frame.shape != (2, 3, 256, 256):
                raise RuntimeError(f"Unexpected frame shape {frame.shape} at episode {episode_id} start {start}.")
            if state.ndim != 2 or int(state.shape[0]) != 2:
                raise RuntimeError(f"Unexpected state shape {state.shape} at episode {episode_id} start {start}.")
            if action.shape != (stride, action.shape[-1]):
                raise RuntimeError(f"Unexpected action shape {action.shape} at episode {episode_id} start {start}.")
            frames.append(np.ascontiguousarray(frame, dtype=np.uint8))
            proprio.append(np.ascontiguousarray(state[0], dtype=np.float32))
            actions.append(np.ascontiguousarray(action, dtype=np.float32).reshape(-1))
            episode_ids.append(episode_id)
            base_indices.append(int(start))
        print(f"[data] episode {episode_id} length {int(length)} windows {len(starts)}", flush=True)

    if not frames:
        raise RuntimeError("No valid photograph windows were collected.")
    fit_ids, test_ids = split_episodes(usable, args.fit_ratio)
    return {
        "frames": np.stack(frames, axis=0),
        "proprio": np.stack(proprio, axis=0),
        "action": np.stack(actions, axis=0),
        "episode_id": np.asarray(episode_ids, dtype=np.int32),
        "base_index": np.asarray(base_indices, dtype=np.int32),
        "fit_episodes": np.asarray(fit_ids, dtype=np.int32),
        "test_episodes": np.asarray(test_ids, dtype=np.int32),
        "stride": int(stride),
        "fps": float(dataset.action_hz),
    }


def cache_signature(bundle: dict, args: argparse.Namespace) -> dict:
    return {
        "dataset": getattr(args, "dataset_name", DATASET_NAME),
        "frame_dt_sec": float(args.frame_dt_sec),
        "fps": float(bundle["fps"]),
        "stride": int(bundle["stride"]),
        "windows_per_episode": int(args.windows_per_episode),
        "fit_ratio": float(args.fit_ratio),
        "fit_episodes": [int(v) for v in bundle["fit_episodes"]],
        "test_episodes": [int(v) for v in bundle["test_episodes"]],
        "n_windows": int(bundle["frames"].shape[0]),
    }


def save_windows(path: Path, bundle: dict, signature: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        frames=bundle["frames"],
        proprio=bundle["proprio"],
        action=bundle["action"],
        episode_id=bundle["episode_id"],
        base_index=bundle["base_index"],
        fit_episodes=bundle["fit_episodes"],
        test_episodes=bundle["test_episodes"],
    )
    path.with_suffix(".json").write_text(json.dumps(signature, indent=2), encoding="utf-8")


def load_windows(path: Path, signature: dict) -> dict | None:
    meta_path = path.with_suffix(".json")
    if not path.exists() or not meta_path.exists():
        return None
    saved = json.loads(meta_path.read_text(encoding="utf-8"))
    comparable = {key: signature[key] for key in signature if key != "n_windows"}
    saved_comparable = {key: saved.get(key) for key in comparable}
    if saved_comparable != comparable:
        print("[data] existing window cache does not match this run; recomputing.", flush=True)
        return None
    loaded = np.load(path)
    bundle = {key: loaded[key] for key in loaded.files}
    bundle["stride"] = int(saved["stride"])
    bundle["fps"] = float(saved["fps"])
    print(f"[data] reused {bundle['frames'].shape[0]} cached windows from {path}", flush=True)
    return bundle


def extract_latents(name: str, ckpt: Path, bundle: dict, args: argparse.Namespace, out_path: Path) -> dict:
    if not args.recompute and out_path.exists():
        loaded = np.load(out_path)
        print(f"[idm] reused cached {name} latents from {out_path}", flush=True)
        return {"z": loaded["z"], "visual": loaded["visual"]}

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this process. Run the script as root on a free GPU, "
            "or pass --device cpu."
        )
    print(f"[idm] loading {name} from {ckpt}", flush=True)
    model = load_latent_action_model(str(ckpt), str(RELEASE_YAML))
    device = torch.device(args.device)
    model = model.to(device).eval()
    frames = bundle["frames"]
    z_parts = []
    visual_parts = []
    batch_size = max(1, int(args.batch_size))
    for start in range(0, frames.shape[0], batch_size):
        batch_np = np.array(frames[start : start + batch_size], copy=True)
        batch = torch.from_numpy(batch_np).to(device, non_blocking=True)
        video, _ = gpu_two_view_video_aug(batch, training=False)
        with torch.inference_mode():
            out = model.get_latent_action(
                videos=video,
                states=None,
                dec_videos=video,
                predict_future_frame=False,
            )
            features = model.extract_vision_features(video[:, :1])
        z = out["quantized"].detach().float().reshape(video.shape[0], -1).cpu().numpy()
        if features.ndim == 4:
            visual = features.detach().float().mean(dim=2).squeeze(1)
        elif features.ndim == 3:
            visual = features.detach().float().mean(dim=1)
        else:
            raise RuntimeError(f"Unexpected vision feature shape {tuple(features.shape)}.")
        z_parts.append(z)
        visual_parts.append(visual.cpu().numpy())
        done = min(start + batch_size, frames.shape[0])
        if done == frames.shape[0] or done % (batch_size * 20) == 0:
            print(f"[idm] {name} {done}/{frames.shape[0]}", flush=True)
    result = {
        "z": np.concatenate(z_parts, axis=0).astype(np.float32),
        "visual": np.concatenate(visual_parts, axis=0).astype(np.float32),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _standardize_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def _apply_standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values.astype(np.float64) - mean) / std


def _solve_ridge(x_values: np.ndarray, y_values: np.ndarray, alpha: float) -> np.ndarray:
    dim = x_values.shape[1]
    gram = x_values.T @ x_values
    gram.flat[:: dim + 1] += float(alpha)
    cross = x_values.T @ y_values
    try:
        return np.linalg.solve(gram, cross)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, cross, rcond=None)[0]


def _r2(y_values: np.ndarray, prediction: np.ndarray) -> float:
    residual = float(np.sum((y_values - prediction) ** 2))
    baseline = float(np.sum(y_values ** 2))
    if baseline <= 0.0:
        return float("nan")
    return float(1.0 - residual / baseline)


def _mean_dim_r2(y_values: np.ndarray, prediction: np.ndarray) -> float:
    residual = np.sum((y_values - prediction) ** 2, axis=0)
    baseline = np.sum(y_values ** 2, axis=0)
    valid = baseline > 1e-8
    if not np.any(valid):
        return float("nan")
    return float(np.mean(1.0 - residual[valid] / baseline[valid]))


def _fit_standardized(
    x_values: np.ndarray,
    y_values: np.ndarray,
    fit_mask: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean, x_std = _standardize_fit(x_values[fit_mask])
    y_mean, y_std = _standardize_fit(y_values[fit_mask])
    weight = _solve_ridge(
        _apply_standardize(x_values[fit_mask], x_mean, x_std),
        _apply_standardize(y_values[fit_mask], y_mean, y_std),
        alpha,
    )
    return x_mean, x_std, y_mean, y_std, weight


def _predict(
    x_values: np.ndarray,
    y_values: np.ndarray,
    mask: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _apply_standardize(y_values[mask], y_mean, y_std),
        _apply_standardize(x_values[mask], x_mean, x_std) @ weight,
    )


def fit_ridge(
    x_values: np.ndarray,
    y_values: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict:
    chosen = float(ALPHAS[len(ALPHAS) // 2])
    val_r2 = None
    if int(val_mask.sum()) >= 2 and int(train_mask.sum()) >= 2:
        best = None
        for alpha in ALPHAS:
            x_mean, x_std, y_mean, y_std, weight = _fit_standardized(x_values, y_values, train_mask, alpha)
            y_val, y_hat = _predict(x_values, y_values, val_mask, x_mean, x_std, y_mean, y_std, weight)
            score = _r2(y_val, y_hat)
            if best is None or score > best[0]:
                best = (score, float(alpha))
        val_r2, chosen = best

    fit_mask = train_mask | val_mask
    x_mean, x_std, y_mean, y_std, weight = _fit_standardized(x_values, y_values, fit_mask, chosen)
    y_fit, fit_prediction = _predict(x_values, y_values, fit_mask, x_mean, x_std, y_mean, y_std, weight)
    y_test, prediction = _predict(x_values, y_values, test_mask, x_mean, x_std, y_mean, y_std, weight)
    return {
        "alpha": chosen,
        "val_r2": val_r2,
        "fit_r2": _r2(y_fit, fit_prediction),
        "test_r2": _r2(y_test, prediction),
        "test_mean_dim_r2": _mean_dim_r2(y_test, prediction),
        "n_fit": int(fit_mask.sum()),
        "n_test": int(test_mask.sum()),
        "x_dim": int(x_values.shape[1]),
        "y_dim": int(y_values.shape[1]),
    }


def masks_for(bundle: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_ids = set(int(v) for v in bundle["fit_episodes"])
    episode_ids = bundle["episode_id"].astype(int)
    fit_mask = np.asarray([int(episode_id) in fit_ids for episode_id in episode_ids])
    ordered_fit = [int(v) for v in bundle["fit_episodes"]]
    n_inner = max(1, int(round(len(ordered_fit) * 0.2)))
    n_inner = min(n_inner, len(ordered_fit) - 1)
    inner_ids = set(ordered_fit[-n_inner:])
    val_mask = np.asarray([int(episode_id) in inner_ids for episode_id in episode_ids])
    train_mask = fit_mask & ~val_mask
    test_mask = ~fit_mask
    return train_mask, val_mask, test_mask


def mean_cosine(lhs: np.ndarray, rhs: np.ndarray) -> float:
    left = lhs.astype(np.float64)
    right = rhs.astype(np.float64)
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = denom > 1e-8
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sum(left[valid] * right[valid], axis=1) / denom[valid]))


def round_result(result: dict) -> dict:
    rendered = {}
    for key, value in result.items():
        if isinstance(value, float):
            rendered[key] = None if np.isnan(value) else round(value, 6)
        else:
            rendered[key] = value
    return rendered


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    window_path = args.output_dir / "photograph_windows.npz"
    dataset, stride = build_dataset(args)
    print(
        f"[data] {DATASET_NAME} episodes {len(dataset.trajectory_ids)} "
        f"fps {float(dataset.action_hz):.3f} stride {stride}",
        flush=True,
    )
    bundle = None
    if not args.recompute:
        signature_hint = {
            "dataset": DATASET_NAME,
            "frame_dt_sec": float(args.frame_dt_sec),
            "fps": float(dataset.action_hz),
            "stride": int(stride),
            "windows_per_episode": int(args.windows_per_episode),
            "fit_ratio": float(args.fit_ratio),
        }
        meta_path = window_path.with_suffix(".json")
        if meta_path.exists():
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
            if all(saved.get(key) == value for key, value in signature_hint.items()):
                bundle = load_windows(
                    window_path,
                    {
                        **signature_hint,
                        "fit_episodes": saved["fit_episodes"],
                        "test_episodes": saved["test_episodes"],
                        "n_windows": saved["n_windows"],
                    },
                )
    if bundle is None:
        bundle = collect_windows(args, dataset, stride)
        save_windows(window_path, bundle, cache_signature(bundle, args))
        print(f"[data] saved windows to {window_path}", flush=True)
    del dataset

    models = {
        "release": extract_latents("release", RELEASE_CKPT, bundle, args, args.output_dir / "release_latents.npz"),
        "C": extract_latents("C", C_CKPT, bundle, args, args.output_dir / "c_latents.npz"),
    }
    train_mask, val_mask, test_mask = masks_for(bundle)
    action = bundle["action"].astype(np.float32)
    proprio = bundle["proprio"].astype(np.float32)
    proprio_result = fit_ridge(proprio, action, train_mask, val_mask, test_mask)
    report = {
        "dataset": DATASET_NAME,
        "frame_dt_sec": float(args.frame_dt_sec),
        "fps": float(bundle["fps"]),
        "stride": int(bundle["stride"]),
        "action_dim": int(action.shape[1]),
        "n_fit_episodes": int(bundle["fit_episodes"].shape[0]),
        "n_test_episodes": int(bundle["test_episodes"].shape[0]),
        "n_fit_windows": int((train_mask | val_mask).sum()),
        "n_test_windows": int(test_mask.sum()),
        "fit_episodes": [int(v) for v in bundle["fit_episodes"]],
        "test_episodes": [int(v) for v in bundle["test_episodes"]],
        "proprio_to_action": round_result(proprio_result),
        "models": {},
        "latent_cosine_release_vs_C": round(mean_cosine(models["release"]["z"], models["C"]["z"]), 6),
    }
    for name, features in models.items():
        latent_result = fit_ridge(features["z"], action, train_mask, val_mask, test_mask)
        visual_result = fit_ridge(features["visual"], features["z"], train_mask, val_mask, test_mask)
        report["models"][name] = {
            "ckpt": str(RELEASE_CKPT if name == "release" else C_CKPT),
            "latent_to_action": round_result(latent_result),
            "current_visual_to_latent": round_result(visual_result),
            "beats_proprio": bool(latent_result["test_r2"] > proprio_result["test_r2"]),
        }
    result_path = args.output_dir / "ridge_result.json"
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[ridge] photograph held-out R² (higher means the frozen code linearly explains the target)", flush=True)
    print(
        f"[ridge] proprio → action                 test R² {proprio_result['test_r2']:.4f}",
        flush=True,
    )
    for name in ("release", "C"):
        item = report["models"][name]
        print(
            f"[ridge] {name:8s} latent → action          test R² {item['latent_to_action']['test_r2']:.4f}    "
            f"fit R² {item['latent_to_action']['fit_r2']:.4f}",
            flush=True,
        )
        print(
            f"[ridge] {name:8s} current frame → latent   test R² {item['current_visual_to_latent']['test_r2']:.4f}",
            flush=True,
        )
    print(f"[ridge] cosine(release z, C z) {report['latent_cosine_release_vs_C']:.4f}", flush=True)
    print(f"[ridge] wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
