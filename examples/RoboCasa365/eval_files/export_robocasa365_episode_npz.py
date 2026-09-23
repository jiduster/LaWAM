from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from examples.RoboCasa365.eval_files.robocasa365_common import (
    POLICY_ACTION_ORDER,
    POLICY_STATE_ORDER,
    RAW_ACTION_ORDER,
    RAW_STATE_ORDER,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    if chunks_size <= 0:
        return 0
    return int(episode_index) // int(chunks_size)


def _load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    return json.loads(info_path.read_text(encoding="utf-8"))


def _load_episode_row(dataset_root: Path, episode_index: int) -> dict[str, Any]:
    rows = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    for row in rows:
        if int(row.get("episode_index", -1)) == int(episode_index):
            return row
    raise KeyError(f"Episode index {episode_index} not found in {dataset_root / 'meta' / 'episodes.jsonl'}.")


def _parquet_path(dataset_root: Path, episode_index: int, info: dict[str, Any]) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    rel = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
    path = rel.format(
        episode_chunk=_episode_chunk(episode_index, chunks_size),
        episode_index=int(episode_index),
    )
    return dataset_root / path


def _video_paths(dataset_root: Path, episode_index: int, info: dict[str, Any]) -> dict[str, str]:
    chunks_size = int(info.get("chunks_size", 1000))
    rel = str(info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"))
    out = {}
    for key in ("robot0_agentview_right", "robot0_eye_in_hand", "robot0_agentview_left"):
        video_key = f"observation.images.{key}"
        path = rel.format(
            episode_chunk=_episode_chunk(episode_index, chunks_size),
            episode_index=int(episode_index),
            video_key=video_key,
        )
        out[key] = str(dataset_root / path)
    return out


def _reorder_columns(values: np.ndarray, src_order: list[str], dst_order: list[str]) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(src_order)}
    cols = [index[name] for name in dst_order]
    return np.asarray(values[:, cols], dtype=np.float32)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--episode_index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    episode_index = int(args.episode_index)
    info = _load_info(dataset_root)
    episode = _load_episode_row(dataset_root, episode_index)
    parquet_path = _parquet_path(dataset_root, episode_index, info)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    table = pq.read_table(parquet_path)
    data = table.to_pydict()
    raw_action = np.asarray(data["action"], dtype=np.float32)
    raw_state = np.asarray(data["observation.state"], dtype=np.float32)
    timestamps = np.asarray(data["timestamp"], dtype=np.float32)
    frame_index = np.asarray(data["frame_index"], dtype=np.int64)
    episode_index_col = np.asarray(data["episode_index"], dtype=np.int64)
    if raw_action.ndim != 2 or raw_action.shape[1] != 12:
        raise ValueError(f"Expected raw action shape [T,12], got {raw_action.shape}.")
    if raw_state.ndim != 2 or raw_state.shape[1] != 16:
        raise ValueError(f"Expected raw state shape [T,16], got {raw_state.shape}.")
    if not np.all(episode_index_col == episode_index):
        raise ValueError(f"Parquet {parquet_path} contains rows from a different episode.")

    policy_action = _reorder_columns(raw_action, RAW_ACTION_ORDER, POLICY_ACTION_ORDER)
    policy_state = _reorder_columns(raw_state, RAW_STATE_ORDER, POLICY_STATE_ORDER)
    tasks = episode.get("tasks") or []
    prompt = str(tasks[0] if tasks else "")
    extras_dir = dataset_root / "extras" / f"episode_{episode_index:06d}"
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    video_paths = _video_paths(dataset_root, episode_index, info)
    metadata = {
        "dataset_root": str(dataset_root),
        "episode_index": episode_index,
        "length": int(raw_action.shape[0]),
        "fps": float(info.get("fps", 20.0)),
        "prompt": prompt,
        "raw_action_order": RAW_ACTION_ORDER,
        "policy_action_order": POLICY_ACTION_ORDER,
        "raw_state_order": RAW_STATE_ORDER,
        "policy_state_order": POLICY_STATE_ORDER,
        "parquet_path": str(parquet_path),
        "extras_dir": str(extras_dir),
        "model_xml_gz": str(extras_dir / "model.xml.gz"),
        "states_npz": str(extras_dir / "states.npz"),
        "ep_meta_json": str(extras_dir / "ep_meta.json"),
        "video_paths": video_paths,
    }
    np.savez_compressed(
        output,
        raw_action_le_order=raw_action,
        policy_action_order=policy_action,
        raw_state_le_order=raw_state,
        policy_state_order=policy_state,
        timestamps=timestamps,
        frame_index=frame_index,
        prompt=np.asarray(prompt),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
