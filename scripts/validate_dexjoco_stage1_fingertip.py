"""Validate Stage 1 DexJoCo fingertip sidecars and temporal alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def validate_sidecars(sidecar_root: Path, dataset_name: str, expected_points: int = 8) -> None:
    dataset_root = sidecar_root / dataset_name
    episode_dir = dataset_root / "episodes"
    files = sorted(episode_dir.glob("episode-*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode sidecars found under {episode_dir}")

    total_frames = 0
    for path in files:
        with np.load(path, allow_pickle=False) as sidecar:
            required = {"tip_position", "valid_mask", "frame_index", "global_index"}
            missing = sorted(required.difference(sidecar.files))
            if missing:
                raise KeyError(f"{path} is missing keys: {missing}")
            positions = np.asarray(sidecar["tip_position"])
            valid = np.asarray(sidecar["valid_mask"])
            frame_index = np.asarray(sidecar["frame_index"])
            if positions.ndim != 3 or positions.shape[1:] != (expected_points, 3):
                raise ValueError(f"{path}: expected [T,{expected_points},3], got {positions.shape}")
            if valid.shape != positions.shape[:2]:
                raise ValueError(f"{path}: valid_mask shape {valid.shape} != {positions.shape[:2]}")
            if frame_index.shape != (positions.shape[0],):
                raise ValueError(f"{path}: frame_index shape {frame_index.shape}")
            if not np.isfinite(positions).all():
                raise ValueError(f"{path}: tip_position contains non-finite values")
            total_frames += positions.shape[0]

    # Stage 1 uses the existing raw sidecars online. The actual target is
    # computed by the collator from endpoints separated by 1.6 seconds.
    print(
        f"Validated {len(files)} episodes, {total_frames} frames, "
        f"{expected_points} fingertip points per frame."
    )
    print("Stage 1 target: tip_position[t + round(1.6 * fps)] - tip_position[t]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-points", type=int, default=8)
    args = parser.parse_args()
    validate_sidecars(args.sidecar_root, args.dataset, args.num_points)


if __name__ == "__main__":
    main()
