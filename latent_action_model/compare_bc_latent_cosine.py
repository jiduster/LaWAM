"""Cosine between frozen B-group and C-group IDM latents.

Both models see the same ego frame pair. The fingertip head is not loaded.
A cosine near 1 means the Stage 1 fingertip loss did not move the latent that
Stage 2 actually consumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_action_model.core.lam_model import load_latent_action_model
from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug
from latent_action_model.eval_idm_action_ridge import (
    RELEASE_YAML,
    build_dataset,
    cache_signature,
    collect_windows,
    load_windows,
    save_windows,
)


B_CKPT = (
    ROOT
    / "latent_action_model/logs/dino_large_vae_dexjoco_bimanual_exclude_photograph_stage1/pytorch_model.pt"
)
C_CKPT = (
    ROOT
    / "latent_action_model/logs/dino_large_vae_dexjoco_bimanual_exclude_photograph_fingertip_stage1/pytorch_model.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="bimanual_photograph")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "latent_action_model/logs/idm_bc_cosine",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--windows-per-episode", type=int, default=8)
    parser.add_argument("--frame-dt-sec", type=float, default=1.6)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    args.fit_ratio = 1.0
    return args


def extract_z(name: str, ckpt: Path, bundle: dict, args: argparse.Namespace, out_path: Path) -> np.ndarray:
    if not args.recompute and out_path.exists():
        loaded = np.load(out_path)
        print(f"[idm] reused cached {name} latents from {out_path}", flush=True)
        return loaded["z"]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this process. Run this command as root on a free GPU."
        )
    print(f"[idm] loading {name} from {ckpt}", flush=True)
    model = load_latent_action_model(str(ckpt), str(RELEASE_YAML)).to(args.device).eval()
    frames = bundle["frames"]
    parts = []
    batch_size = max(1, int(args.batch_size))
    for start in range(0, frames.shape[0], batch_size):
        batch_np = np.array(frames[start : start + batch_size], copy=True)
        batch = torch.from_numpy(batch_np).to(args.device, non_blocking=True)
        video, _ = gpu_two_view_video_aug(batch, training=False)
        with torch.inference_mode():
            out = model.get_latent_action(
                videos=video,
                states=None,
                dec_videos=video,
                predict_future_frame=False,
            )
        parts.append(out["quantized"].detach().float().reshape(video.shape[0], -1).cpu().numpy())
        done = min(start + batch_size, frames.shape[0])
        if done == frames.shape[0] or done % (batch_size * 20) == 0:
            print(f"[idm] {name} {done}/{frames.shape[0]}", flush=True)
    z = np.concatenate(parts, axis=0).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, z=z)
    del model
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return z


def row_cosine(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(lhs, axis=1) * np.linalg.norm(rhs, axis=1)
    cosine = np.sum(lhs * rhs, axis=1) / np.maximum(denom, 1e-8)
    return cosine.astype(np.float64)


def summarize(cosine: np.ndarray, lhs: np.ndarray, rhs: np.ndarray) -> dict:
    delta = np.linalg.norm(lhs - rhs, axis=1)
    base = np.maximum(np.linalg.norm(lhs, axis=1), 1e-8)
    relative_l2 = delta / base
    return {
        "n_windows": int(cosine.shape[0]),
        "latent_dim": int(lhs.shape[1]),
        "cosine_mean": round(float(cosine.mean()), 6),
        "cosine_median": round(float(np.median(cosine)), 6),
        "cosine_p10": round(float(np.percentile(cosine, 10)), 6),
        "cosine_p90": round(float(np.percentile(cosine, 90)), 6),
        "fraction_cosine_above_0.95": round(float(np.mean(cosine > 0.95)), 6),
        "fraction_cosine_above_0.99": round(float(np.mean(cosine > 0.99)), 6),
        "relative_l2_mean": round(float(relative_l2.mean()), 6),
        "relative_l2_median": round(float(np.median(relative_l2)), 6),
    }


def load_or_collect(args: argparse.Namespace) -> dict:
    window_path = args.output_dir / f"{args.dataset_name}_windows.npz"
    dataset, stride = build_dataset(args)
    print(
        f"[data] {args.dataset_name} episodes {len(dataset.trajectory_ids)} "
        f"fps {float(dataset.action_hz):.3f} stride {stride}",
        flush=True,
    )
    bundle = None
    if not args.recompute and window_path.with_suffix(".json").exists():
        saved = json.loads(window_path.with_suffix(".json").read_text(encoding="utf-8"))
        bundle = load_windows(window_path, saved)
    if bundle is None:
        bundle = collect_windows(args, dataset, stride)
        save_windows(window_path, bundle, cache_signature(bundle, args))
        print(f"[data] saved windows to {window_path}", flush=True)
    return bundle


def main() -> None:
    args = parse_args()
    if not B_CKPT.is_file() or not C_CKPT.is_file():
        raise FileNotFoundError(f"Missing checkpoint: B={B_CKPT.exists()} C={C_CKPT.exists()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_or_collect(args)
    z_b = extract_z("B", B_CKPT, bundle, args, args.output_dir / f"{args.dataset_name}_B.npz")
    z_c = extract_z("C", C_CKPT, bundle, args, args.output_dir / f"{args.dataset_name}_C.npz")
    if z_b.shape != z_c.shape:
        raise RuntimeError(f"Latent shape mismatch: B {z_b.shape} vs C {z_c.shape}")
    cosine = row_cosine(z_b, z_c)
    report = {
        "dataset": args.dataset_name,
        "frame_dt_sec": float(args.frame_dt_sec),
        "stride": int(bundle["stride"]),
        "windows_per_episode": int(args.windows_per_episode),
        "B_ckpt": str(B_CKPT),
        "C_ckpt": str(C_CKPT),
        **summarize(cosine, z_b, z_c),
    }
    out_path = args.output_dir / f"{args.dataset_name}_bc_cosine.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"\n[cosine] {args.dataset_name}  B vs C"
        f"\n[cosine] mean {report['cosine_mean']:.4f}   median {report['cosine_median']:.4f}"
        f"\n[cosine] p10 {report['cosine_p10']:.4f}   p90 {report['cosine_p90']:.4f}"
        f"\n[cosine] fraction >0.95 {report['fraction_cosine_above_0.95']:.3f}"
        f"    >0.99 {report['fraction_cosine_above_0.99']:.3f}"
        f"\n[cosine] relative L2 mean {report['relative_l2_mean']:.4f}",
        flush=True,
    )
    print(f"[cosine] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
