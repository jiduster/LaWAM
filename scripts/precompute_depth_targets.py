"""Precompute MoGe-2 + LingBot-Depth features for LaWAM depth queries.

The photograph VLM sees the current ego / wrist_left / wrist_right frame after the
same 256 resize used in training. This script runs those frames through the frozen
teachers and writes one bf16 tensor per episode:

    episode_XXXXXX.pt    shape [T, 3, 256, 1024]

`/mnt/ceph/ckpt/lingbot-vla-4b-depth` is the depth-enabled VLA, not the teacher.
The teacher weights are MoGe-2 and LingBot-Depth-Pretrain.

Run this inside a GPU allocation with the lawam environment. That environment is
Python 3.10 and already has pytorch3d (installed as pipablepytorch3d). MoGe and
LingBot-Depth are imported from the local source trees, not installed into lawam.

    cd /data/home/zyh/LaWAM
    /data/home/zyh/miniconda3/envs/lawam/bin/python scripts/precompute_depth_targets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.model.framework.vlas.depth_queries import (
    DEPTH_TARGET_DIM,
    DEPTH_TARGET_TOKENS,
    DEPTH_VIEW_NAMES,
    FrozenDepthTeacher,
)


VIEW_KEYS = tuple(f"video.{name}" for name in DEPTH_VIEW_NAMES)
DEFAULT_OUTPUT = "/mnt/ceph3/zyh/LaWAM_depth_targets/dexjoco_bimanual_photograph"
DEFAULT_MOGE = "/mnt/ceph/ckpt/MoGe2/MoGe_2_vit_b_normal.pt"
DEFAULT_MORGBD = "/mnt/ceph/ckpt/lingbot-depth-pretrain-vitl-14/model.pt"
DEFAULT_DATA_ROOT = "/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets"


def _current_chw(value: torch.Tensor, *, key: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 4 and int(tensor.shape[1]) == 3:
        frame = tensor[0]
    elif tensor.ndim == 3 and int(tensor.shape[0]) == 3:
        frame = tensor
    else:
        raise ValueError(f"{key} frame must be [T,3,H,W] or [3,H,W], got {tuple(tensor.shape)}.")
    if tuple(frame.shape[-2:]) != (256, 256):
        raise ValueError(f"{key} frame is {tuple(frame.shape)}, expected 3x256x256 after the training resize.")
    return frame.contiguous()


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset-name", default="bimanual_photograph")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--moge-path", default=DEFAULT_MOGE)
    parser.add_argument("--morgbd-path", default=DEFAULT_MORGBD)
    parser.add_argument("--batch-frames", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this script inside a GPU allocation.")
    for path in (args.moge_path, args.morgbd_path):
        if not Path(path).is_file():
            raise SystemExit(f"Teacher weight file not found: {path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset_name,
        "views": list(DEPTH_VIEW_NAMES),
        "tokens": DEPTH_TARGET_TOKENS,
        "channels": DEPTH_TARGET_DIM,
        "image_hw": [256, 256],
        "dtype": "bfloat16",
        "moge_path": str(args.moge_path),
        "morgbd_path": str(args.morgbd_path),
        "complete": False,
    }
    _write_manifest(output_dir / "manifest.json", manifest)

    data_cfg = OmegaConf.create(
        {
            "image_resolution": 256,
            "num_frames": 1,
            "video_backend": "pyav",
            "enable_video_frame_cache": False,
            "val_tail_ratio": 0.0,
        }
    )
    dataset = make_LeRobotSingleDataset(
        data_root_dir=args.data_root,
        data_name=args.dataset_name,
        robot_type="dexjoco_bimanual",
        mode="all",
        data_cfg=data_cfg,
        framework_name="LaWAM",
    )
    dataset.transforms.eval()

    print(
        f"Loading frozen teachers MoGe={args.moge_path} LingBot-Depth={args.morgbd_path}",
        flush=True,
    )
    teacher = FrozenDepthTeacher(moge_path=args.moge_path, morgbd_path=args.morgbd_path).to(torch.device(args.device))

    episode_ids = [int(item) for item in dataset._trajectory_ids.tolist()]
    episode_lengths = [int(item) for item in dataset._trajectory_lengths.tolist()]
    print(f"Episodes={len(episode_ids)} frames={sum(episode_lengths)}", flush=True)

    for episode_index, length in zip(episode_ids, episode_lengths):
        destination = output_dir / f"episode_{episode_index:06d}.pt"
        if destination.is_file():
            existing = torch.load(destination, map_location="cpu", weights_only=True)
            if tuple(existing.shape) == (length, len(DEPTH_VIEW_NAMES), DEPTH_TARGET_TOKENS, DEPTH_TARGET_DIM):
                print(f"skip episode {episode_index} ({length} frames)", flush=True)
                continue
        chunks: list[torch.Tensor] = []
        pending: list[torch.Tensor] = []

        def flush() -> None:
            if not pending:
                return
            batch = torch.stack(pending, dim=0)
            features = teacher.features(batch.to(args.device))
            features = features.view(batch.shape[0], len(DEPTH_VIEW_NAMES), DEPTH_TARGET_TOKENS, DEPTH_TARGET_DIM)
            chunks.append(features.to(dtype=torch.bfloat16, device="cpu"))
            pending.clear()

        for frame_index in range(length):
            raw = dataset.get_step_data(episode_index, frame_index)
            transformed = dataset.transforms(raw)
            views = [_current_chw(transformed[key], key=key) for key in VIEW_KEYS]
            pending.append(torch.stack(views, dim=0))
            if len(pending) >= int(args.batch_frames):
                flush()
        flush()
        episode_features = torch.cat(chunks, dim=0).contiguous()
        if tuple(episode_features.shape) != (length, len(DEPTH_VIEW_NAMES), DEPTH_TARGET_TOKENS, DEPTH_TARGET_DIM):
            raise RuntimeError(
                f"Episode {episode_index} feature shape {tuple(episode_features.shape)} "
                f"does not match length {length}."
            )
        torch.save(episode_features, destination)
        print(f"wrote {destination} {tuple(episode_features.shape)}", flush=True)

    manifest["complete"] = True
    manifest["episodes"] = episode_ids
    _write_manifest(output_dir / "manifest.json", manifest)
    print("depth queries target preprocess complete", flush=True)


if __name__ == "__main__":
    main()
