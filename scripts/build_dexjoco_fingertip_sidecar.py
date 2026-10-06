#!/usr/bin/env python3
"""Extract DexJoCo Allegro fingertip labels from the existing DexWM FK cache.

The DexWM preparation pipeline already computes 21 FK points per hand from the
official DexJoCo XML. This converter keeps only the four real fingertips per
hand and stores them as a training-only sidecar, without changing the LeRobot
parquet schema.

Output layout::

    <output-root>/<dataset-name>/manifest.json
    <output-root>/<dataset-name>/metadata.json
    <output-root>/<dataset-name>/episodes/episode-000000.npz

The exported coordinates are the fixed DexJoCo ego-camera optical coordinates
in meters. They are suitable for the fixed-camera DexJoCo data used here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# DexWM's 21-point layout is [palm, thumb(4), ff(4), mf(4), rf(4), rf-proxy(4)].
# Keep only [ff distal, mf distal, rf distal, thumb distal] for each hand.
TIP_NAMES = (
    "left_ff",
    "left_mf",
    "left_rf",
    "left_th",
    "right_ff",
    "right_mf",
    "right_rf",
    "right_th",
)
TIP_INDICES = np.asarray([5 + 3, 5 + 7, 5 + 11, 1 + 3, 21 + 5 + 3, 21 + 5 + 7, 21 + 5 + 11, 21 + 1 + 3], dtype=np.int64)


def _read_lerobot_rows(dataset_root: Path) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    files = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot data parquet files found under {dataset_root / 'data'}")
    tables = [pq.read_table(path, columns=["episode_index", "frame_index", "index"]) for path in files]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    rows = table.to_pydict()
    episode = np.asarray(rows["episode_index"], dtype=np.int64)
    frame = np.asarray(rows["frame_index"], dtype=np.int64)
    global_index = np.asarray(rows["index"], dtype=np.int64)
    result: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for episode_id in np.unique(episode):
        mask = episode == int(episode_id)
        result[int(episode_id)] = (frame[mask], global_index[mask], np.flatnonzero(mask))
    return result


def _load_source_episode(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        required = {"points", "episode_index", "frame_index", "global_index"}
        missing = sorted(required.difference(source.files))
        if missing:
            raise KeyError(f"Source cache {path} is missing keys: {missing}")
        episode = np.asarray(source["episode_index"], dtype=np.int64)
        frame = np.asarray(source["frame_index"], dtype=np.int64)
        global_index = np.asarray(source["global_index"], dtype=np.int64)
        points = np.asarray(source["points"], dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (42, 3):
        raise ValueError(f"Expected source points [T,42,3] in {path}, got {points.shape}")
    if not (len(episode) == len(frame) == len(global_index) == len(points)):
        raise ValueError(f"Source cache arrays have inconsistent lengths in {path}")
    return episode, frame, global_index, points


def main(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dataset_name = str(args.dataset_name or dataset_root.name)
    source_root = Path(args.source_wm_cache_root).expanduser().resolve() / dataset_name
    output_dataset_root = Path(args.output_root).expanduser().resolve() / dataset_name
    output_episode_root = output_dataset_root / "episodes"
    output_episode_root.mkdir(parents=True, exist_ok=True)

    parquet_rows = _read_lerobot_rows(dataset_root)
    manifest_path = source_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing DexWM source manifest: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text())
    source_episodes = source_manifest.get("episodes", [])
    if not source_episodes:
        raise ValueError(f"No episodes listed in {manifest_path}")

    output_manifest: list[dict[str, int | str]] = []
    for item in sorted(source_episodes, key=lambda row: int(row["episode_index"])):
        episode_id = int(item["episode_index"])
        source_path = source_root / str(item["file"])
        output_path = output_episode_root / f"episode-{episode_id:06d}.npz"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_path}; pass --overwrite to replace it.")

        source_episode, source_frame, source_global, points = _load_source_episode(source_path)
        if episode_id not in parquet_rows:
            raise KeyError(f"Episode {episode_id} exists in source cache but not in {dataset_root}")
        parquet_frame, parquet_global, _ = parquet_rows[episode_id]
        if not np.array_equal(source_episode, np.full_like(source_episode, episode_id)):
            raise ValueError(f"Source episode ids are inconsistent in {source_path}")
        if not np.array_equal(source_frame, parquet_frame) or not np.array_equal(source_global, parquet_global):
            raise ValueError(
                f"Frame alignment mismatch for episode {episode_id}. "
                "The DexWM cache and LeRobot parquet must come from the same dataset export."
            )

        tip_position = points[:, TIP_INDICES, :].astype(np.float32, copy=False)
        valid_mask = np.isfinite(tip_position).all(axis=-1)
        if not bool(valid_mask.all()):
            invalid = int((~valid_mask).sum())
            raise ValueError(f"Non-finite fingertip labels in episode {episode_id}: {invalid} values")

        tmp_path = output_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp_path,
            episode_index=np.full(len(points), episode_id, dtype=np.int64),
            frame_index=source_frame,
            global_index=source_global,
            tip_position=tip_position,
            valid_mask=valid_mask,
        )
        tmp_path.replace(output_path)
        output_manifest.append({"episode_index": episode_id, "length": int(len(points)), "file": f"episodes/{output_path.name}"})
        print(f"wrote episode {episode_id}: {len(points)} frames", flush=True)

    metadata = {
        "dataset_name": dataset_name,
        "coordinate_frame": "dexjoco_fixed_ego_camera_optical",
        "unit": "meter",
        "tip_names": list(TIP_NAMES),
        "tip_position_shape": [len(TIP_NAMES), 3],
        "source": "DexWM points generated by DexJoCo official XML FK",
    }
    (output_dataset_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dataset_root / "manifest.json").write_text(json.dumps({"episodes": output_manifest}, indent=2) + "\n")
    print(f"Wrote {len(output_manifest)} episode sidecars to {output_dataset_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="LeRobot dataset directory")
    parser.add_argument("--dataset-name", default=None, help="Dataset name; defaults to dataset-root basename")
    parser.add_argument(
        "--source-wm-cache-root",
        required=True,
        help="Root containing <dataset-name>/manifest.json and DexWM episode NPZ files",
    )
    parser.add_argument("--output-root", required=True, help="Root for fingertip sidecars")
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
