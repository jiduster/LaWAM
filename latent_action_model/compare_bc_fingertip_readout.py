"""Frozen linear readout from B/C IDM latents to fingertip displacement.

The IDMs stay frozen. Each gets its own ridge map from its 32-d latent to the
8-fingertip displacement over the same 1.6 s window Stage 1 supervised. The
reported error is held-out RMSE in millimeters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_action_model.compare_bc_latent_cosine import B_CKPT, C_CKPT, extract_z
from latent_action_model.eval_idm_action_ridge import (
    ALPHAS,
    _fit_standardized,
    _predict,
    _r2,
    build_dataset,
    cache_signature,
    collect_windows,
    load_windows,
    masks_for,
    save_windows,
)


SIDECAR_ROOT = Path(
    "/data/home/zyh/LaWAM/latent_action_model/data/dexjoco_fingertip_sidecars_exclude_photograph"
)
TASKS = (
    "bimanual_hanoi",
    "bimanual_assembly",
    "bimanual_microwave_cook",
    "bimanual_unlock_ipad",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "latent_action_model/logs/idm_bc_fingertip_readout",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--windows-per-episode", type=int, default=8)
    parser.add_argument("--fit-ratio", type=float, default=0.7)
    parser.add_argument("--frame-dt-sec", type=float, default=1.6)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    args.dataset_name = args.tasks[0]
    return args


def load_episode_sidecar(dataset_name: str, episode_id: int) -> tuple[np.ndarray, np.ndarray]:
    path = SIDECAR_ROOT / dataset_name / "episodes" / f"episode-{episode_id:06d}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing fingertip sidecar for {dataset_name} episode {episode_id}: {path}")
    with np.load(path) as sidecar:
        positions = np.asarray(sidecar["tip_position"], dtype=np.float32)
        valid = np.asarray(sidecar["valid_mask"], dtype=bool)
    return positions, valid


def fingertip_deltas(dataset_name: str, bundle: dict) -> np.ndarray:
    stride = int(bundle["stride"])
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    deltas = []
    for episode_id, start in zip(bundle["episode_id"], bundle["base_index"]):
        episode_id = int(episode_id)
        start = int(start)
        end = start + stride
        if episode_id not in cache:
            cache[episode_id] = load_episode_sidecar(dataset_name, episode_id)
        positions, valid = cache[episode_id]
        if end >= positions.shape[0]:
            raise RuntimeError(
                f"{dataset_name} episode {episode_id} sidecar length {positions.shape[0]} "
                f"cannot cover start {start} stride {stride}."
            )
        pair_valid = valid[start] & valid[end]
        if not bool(pair_valid.all()):
            raise RuntimeError(
                f"{dataset_name} episode {episode_id} start {start} has invalid fingertips {pair_valid.tolist()}."
            )
        deltas.append((positions[end] - positions[start]).reshape(-1))
    return np.stack(deltas, axis=0).astype(np.float32)


def rmse_mm(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - prediction) ** 2)) * 1000.0)


def score_readout(
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict:
    chosen = float(ALPHAS[len(ALPHAS) // 2])
    if int(val_mask.sum()) >= 2 and int(train_mask.sum()) >= 2:
        best = None
        for alpha in ALPHAS:
            stats = _fit_standardized(features, target, train_mask, alpha)
            _, prediction_std = _predict(features, target, val_mask, *stats)
            y_mean, y_std = stats[2], stats[3]
            prediction = prediction_std * y_std + y_mean
            score = rmse_mm(target[val_mask], prediction)
            if best is None or score < best[0]:
                best = (score, float(alpha))
        chosen = best[1]

    fit_mask = train_mask | val_mask
    x_mean, x_std, y_mean, y_std, weight = _fit_standardized(features, target, fit_mask, chosen)
    y_test_std, prediction_std = _predict(
        features, target, test_mask, x_mean, x_std, y_mean, y_std, weight
    )
    prediction = prediction_std * y_std + y_mean
    y_test = target[test_mask]
    mean_prediction = np.repeat(target[fit_mask].mean(axis=0, keepdims=True), y_test.shape[0], axis=0)
    return {
        "alpha": chosen,
        "test_rmse_mm": round(rmse_mm(y_test, prediction), 4),
        "mean_baseline_rmse_mm": round(rmse_mm(y_test, mean_prediction), 4),
        "test_r2": round(float(_r2(y_test_std, prediction_std)), 4),
        "n_fit": int(fit_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def load_or_collect(args: argparse.Namespace, dataset_name: str) -> tuple[dict, bool]:
    args.dataset_name = dataset_name
    window_path = args.output_dir / dataset_name / "windows.npz"
    dataset, stride = build_dataset(args)
    print(
        f"[data] {dataset_name} episodes {len(dataset.trajectory_ids)} "
        f"fps {float(dataset.action_hz):.3f} stride {stride}",
        flush=True,
    )
    bundle = None
    meta_path = window_path.with_suffix(".json")
    if not args.recompute and meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        hint = {
            "dataset": dataset_name,
            "frame_dt_sec": float(args.frame_dt_sec),
            "windows_per_episode": int(args.windows_per_episode),
            "fit_ratio": float(args.fit_ratio),
        }
        if all(saved.get(key) == value for key, value in hint.items()):
            bundle = load_windows(window_path, saved)
    if bundle is not None:
        return bundle, True
    bundle = collect_windows(args, dataset, stride)
    window_path.parent.mkdir(parents=True, exist_ok=True)
    save_windows(window_path, bundle, cache_signature(bundle, args))
    print(f"[data] saved windows to {window_path}", flush=True)
    return bundle, False


def main() -> None:
    args = parse_args()
    if not B_CKPT.is_file() or not C_CKPT.is_file():
        raise FileNotFoundError(f"Missing checkpoint: B={B_CKPT.exists()} C={C_CKPT.exists()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"tasks": {}}
    for dataset_name in args.tasks:
        bundle, windows_were_cached = load_or_collect(args, dataset_name)
        target_path = args.output_dir / dataset_name / "fingertip_delta.npz"
        if windows_were_cached and target_path.exists():
            target = np.load(target_path)["delta"]
            print(f"[data] reused fingertip deltas {target.shape} for {dataset_name}", flush=True)
        else:
            target = fingertip_deltas(dataset_name, bundle)
            np.savez(target_path, delta=target)
        task_dir = args.output_dir / dataset_name
        z_b = extract_z("B", B_CKPT, bundle, args, task_dir / "B.npz")
        z_c = extract_z("C", C_CKPT, bundle, args, task_dir / "C.npz")
        if z_b.shape[0] != target.shape[0] or z_c.shape[0] != target.shape[0]:
            raise RuntimeError(
                f"Row mismatch for {dataset_name}: B {z_b.shape} C {z_c.shape} delta {target.shape}"
            )
        train_mask, val_mask, test_mask = masks_for(bundle)
        b_score = score_readout(z_b, target, train_mask, val_mask, test_mask)
        c_score = score_readout(z_c, target, train_mask, val_mask, test_mask)
        report["tasks"][dataset_name] = {"B": b_score, "C": c_score}
        print(
            f"[readout] {dataset_name}  "
            f"B {b_score['test_rmse_mm']:.2f} mm   "
            f"C {c_score['test_rmse_mm']:.2f} mm   "
            f"mean-baseline {b_score['mean_baseline_rmse_mm']:.2f} mm   "
            f"R² B {b_score['test_r2']:.3f}  C {c_score['test_r2']:.3f}",
            flush=True,
        )

    b_values = [item["B"]["test_rmse_mm"] for item in report["tasks"].values()]
    c_values = [item["C"]["test_rmse_mm"] for item in report["tasks"].values()]
    report["macro_test_rmse_mm"] = {
        "B": round(float(np.mean(b_values)), 4),
        "C": round(float(np.mean(c_values)), 4),
    }
    out_path = args.output_dir / "fingertip_readout.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"\n[readout] macro RMSE  B {report['macro_test_rmse_mm']['B']:.2f} mm   "
        f"C {report['macro_test_rmse_mm']['C']:.2f} mm",
        flush=True,
    )
    print(f"[readout] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
