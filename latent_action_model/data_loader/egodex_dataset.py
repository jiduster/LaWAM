"""Raw EgoDex dataset reader for Stage 1 LAM training.

EgoDex is distributed as one MP4 and one HDF5 file per demonstration rather
than as a LeRobot dataset. This reader emits the same sample contract as the
LeRobot LAM reader while keeping the original HDF5 annotations intact.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# Stable human-only order: 5 tips per hand. This must not be mixed with the
# 8-point Allegro target used by DexJoCo.
EGODEX_TIP_NAMES: Tuple[str, ...] = (
    "leftThumbTip",
    "leftIndexFingerTip",
    "leftMiddleFingerTip",
    "leftRingFingerTip",
    "leftLittleFingerTip",
    "rightThumbTip",
    "rightIndexFingerTip",
    "rightMiddleFingerTip",
    "rightRingFingerTip",
    "rightLittleFingerTip",
)


@dataclass(frozen=True)
class _EgoDexEpisode:
    task: str
    stem: str
    video_path: Path
    annotation_path: Path
    num_frames: int


class EgoDexLAMDataset(Dataset):
    """Read raw EgoDex MP4/HDF5 demonstrations for LAM continual pretraining."""

    def __init__(
        self,
        data_root_dir: str | Path,
        *,
        split: str = "train",
        tasks: Optional[Sequence[str]] = None,
        num_frames: int = 2,
        frame_dt_sec: float = 0.4,
        fps: float = 30.0,
        image_hw: Tuple[int, int] = (256, 256),
        max_state_dim: int = 14,
        confidence_threshold: float = 0.5,
        frame_stride: int = 1,
        mode: str = "train",
        val_tail_ratio: float = 0.01,
        max_episodes_per_task: Optional[int] = None,
        debug_repeat_batch: bool | int = False,
    ) -> None:
        super().__init__()
        if num_frames != 2:
            raise ValueError(
                "EgoDexLAMDataset currently supports exactly two temporal endpoints; "
                f"got num_frames={num_frames}."
            )
        if frame_dt_sec <= 0 or fps <= 0:
            raise ValueError(f"frame_dt_sec and fps must be positive, got {frame_dt_sec}, {fps}")
        if max_state_dim <= 0:
            raise ValueError(f"max_state_dim must be positive, got {max_state_dim}")
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            )
        if frame_stride <= 0:
            raise ValueError(f"frame_stride must be positive, got {frame_stride}")
        if not 0.0 <= float(val_tail_ratio) < 1.0:
            raise ValueError(f"val_tail_ratio must be in [0, 1), got {val_tail_ratio}")
        mode = str(mode).lower()
        if mode not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unsupported mode={mode!r}")

        self.data_root_dir = Path(data_root_dir).expanduser()
        self.split = str(split)
        self.num_frames = int(num_frames)
        self.frame_dt_sec = float(frame_dt_sec)
        self.fps = float(fps)
        self.temporal_stride = max(1, int(round(self.frame_dt_sec * self.fps)))
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.max_state_dim = int(max_state_dim)
        self.confidence_threshold = float(confidence_threshold)
        self.frame_stride = int(frame_stride)
        self.mode = mode
        self.val_tail_ratio = float(val_tail_ratio)
        self.debug_repeat_batch = debug_repeat_batch

        split_root = self.data_root_dir / self.split
        if not split_root.is_dir():
            raise FileNotFoundError(f"EgoDex split directory does not exist: {split_root}")

        available_tasks = sorted(path.name for path in split_root.iterdir() if path.is_dir())
        if tasks is None or len(tasks) == 0:
            selected_tasks = available_tasks
        else:
            selected_tasks = [str(task) for task in tasks]
            unknown = sorted(set(selected_tasks) - set(available_tasks))
            if unknown:
                raise FileNotFoundError(
                    f"Unknown EgoDex task(s) under {split_root}: {unknown}. "
                    f"Available examples: {available_tasks[:20]}"
                )

        self.episodes: List[_EgoDexEpisode] = []
        for task in selected_tasks:
            task_root = split_root / task
            stems = sorted(
                path.stem
                for path in task_root.glob("*.hdf5")
                if (task_root / f"{path.stem}.mp4").is_file()
            )
            if max_episodes_per_task is not None:
                if max_episodes_per_task <= 0:
                    raise ValueError("max_episodes_per_task must be positive when provided")
                stems = stems[: int(max_episodes_per_task)]
            for stem in stems:
                annotation_path = task_root / f"{stem}.hdf5"
                with h5py.File(annotation_path, "r") as annotations:
                    if "transforms/camera" not in annotations:
                        raise KeyError(f"{annotation_path} is missing transforms/camera")
                    num_frames_in_file = int(annotations["transforms/camera"].shape[0])
                if num_frames_in_file > self.temporal_stride:
                    self.episodes.append(
                        _EgoDexEpisode(
                            task=task,
                            stem=stem,
                            video_path=task_root / f"{stem}.mp4",
                            annotation_path=annotation_path,
                            num_frames=num_frames_in_file,
                        )
                    )

        if not self.episodes:
            raise RuntimeError(
                f"No usable EgoDex episodes found under {split_root} for tasks={selected_tasks}."
            )

        self._episode_window_counts: List[int] = []
        self._episode_window_starts: List[int] = []
        for episode in self.episodes:
            all_count = ((episode.num_frames - self.temporal_stride - 1) // self.frame_stride) + 1
            all_count = max(0, int(all_count))
            val_count = 0
            if self.val_tail_ratio > 0 and all_count > 0:
                val_count = max(1, int(np.ceil(all_count * self.val_tail_ratio)))
                val_count = min(val_count, all_count)
            if mode == "train":
                count = all_count - val_count
                start_rank = 0
            elif mode == "val":
                count = val_count
                start_rank = all_count - val_count
            else:
                count = all_count
                start_rank = 0
            self._episode_window_starts.append(int(start_rank))
            self._episode_window_counts.append(int(count))

        self._cumulative_counts = np.cumsum(self._episode_window_counts, dtype=np.int64)
        if len(self._cumulative_counts) == 0 or int(self._cumulative_counts[-1]) <= 0:
            raise RuntimeError(
                f"No valid {mode} windows for EgoDex tasks={selected_tasks}; "
                f"temporal_stride={self.temporal_stride}, val_tail_ratio={self.val_tail_ratio}."
            )

        # Keep only one annotation episode per worker. The selected EgoDex
        # subset can contain many thousands of videos, so an unbounded cache
        # would eventually exhaust host memory.
        self._annotation_cache_episode_idx: Optional[int] = None
        self._annotation_cache_value: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._video_capture: Optional[cv2.VideoCapture] = None
        self._video_capture_path: Optional[Path] = None
        self._debug_cache: List[Dict] = []
        self._debug_cache_initialized = False

        print(
            f"[EgoDexLAMDataset] split={self.split} mode={self.mode} "
            f"tasks={len(selected_tasks)} episodes={len(self.episodes)} "
            f"windows={len(self)} fps={self.fps:.1f} dt_sec={self.frame_dt_sec:.3f} "
            f"stride={self.temporal_stride} frame_stride={self.frame_stride} "
            f"tips={len(EGODEX_TIP_NAMES)} confidence>={self.confidence_threshold:.2f}",
            flush=True,
        )

    def __len__(self) -> int:
        if self.debug_repeat_batch:
            return 10000
        return int(self._cumulative_counts[-1])

    def _resolve_index(self, index: int) -> Tuple[_EgoDexEpisode, int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_idx = int(bisect.bisect_right(self._cumulative_counts.tolist(), int(index)))
        previous = int(self._cumulative_counts[episode_idx - 1]) if episode_idx > 0 else 0
        rank = int(index) - previous + self._episode_window_starts[episode_idx]
        base_index = rank * self.frame_stride
        return self.episodes[episode_idx], base_index, episode_idx

    def _load_annotations(self, episode_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._annotation_cache_episode_idx == episode_idx and self._annotation_cache_value is not None:
            return self._annotation_cache_value
        episode = self.episodes[episode_idx]
        with h5py.File(episode.annotation_path, "r") as annotations:
            tip_world = np.stack(
                [
                    np.asarray(annotations[f"transforms/{name}"][:, :3, 3], dtype=np.float32)
                    for name in EGODEX_TIP_NAMES
                ],
                axis=1,
            )
            tip_confidence = np.stack(
                [
                    np.asarray(annotations[f"confidences/{name}"], dtype=np.float32)
                    for name in EGODEX_TIP_NAMES
                ],
                axis=1,
            )
            camera_to_world = np.asarray(annotations["transforms/camera"], dtype=np.float32)
        if tip_world.shape != (episode.num_frames, len(EGODEX_TIP_NAMES), 3):
            raise ValueError(
                f"Invalid fingertip shape in {episode.annotation_path}: {tip_world.shape}"
            )
        if tip_confidence.shape != (episode.num_frames, len(EGODEX_TIP_NAMES)):
            raise ValueError(
                f"Invalid confidence shape in {episode.annotation_path}: {tip_confidence.shape}"
            )
        if camera_to_world.shape != (episode.num_frames, 4, 4):
            raise ValueError(
                f"Invalid camera transform in {episode.annotation_path}: {camera_to_world.shape}"
            )
        self._annotation_cache_episode_idx = episode_idx
        self._annotation_cache_value = (tip_world, tip_confidence, camera_to_world)
        return self._annotation_cache_value

    def _read_video(self, episode: _EgoDexEpisode, indices: Sequence[int]) -> torch.Tensor:
        if self._video_capture_path != episode.video_path:
            if self._video_capture is not None:
                self._video_capture.release()
            self._video_capture = cv2.VideoCapture(str(episode.video_path))
            self._video_capture_path = episode.video_path
        if self._video_capture is None or not self._video_capture.isOpened():
            raise RuntimeError(f"Unable to open EgoDex video: {episode.video_path}")

        height, width = self.image_hw
        frames: List[np.ndarray] = []
        for frame_index in indices:
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame_bgr = self._video_capture.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(
                    f"Failed to decode frame {frame_index} from {episode.video_path}"
                )
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            source_h, source_w = frame_rgb.shape[:2]
            scale = max(width / source_w, height / source_h)
            resized_w = max(width, int(round(source_w * scale)))
            resized_h = max(height, int(round(source_h * scale)))
            resized = cv2.resize(frame_rgb, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
            top = (resized_h - height) // 2
            left = (resized_w - width) // 2
            frames.append(np.ascontiguousarray(resized[top : top + height, left : left + width]))
        return torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).contiguous()

    def _make_sample(self, index: int) -> Dict:
        episode, base_index, episode_idx = self._resolve_index(index)
        indices = np.asarray([base_index, base_index + self.temporal_stride], dtype=np.int64)
        tip_world, tip_confidence, camera_to_world = self._load_annotations(episode_idx)

        try:
            camera_world_to_base = np.linalg.inv(camera_to_world[base_index]).astype(np.float32)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Singular camera transform at frame {base_index} in {episode.annotation_path}"
            ) from exc

        homogeneous = np.concatenate(
            [
                tip_world[indices],
                np.ones((len(indices), len(EGODEX_TIP_NAMES), 1), dtype=np.float32),
            ],
            axis=-1,
        )
        # Both endpoints use the starting frame camera coordinates, removing
        # head-camera motion from the physical target.
        tip_positions = np.einsum(
            "ij,tkj->tki", camera_world_to_base, homogeneous
        )[..., :3]
        finite = np.isfinite(tip_positions).all(axis=-1)
        valid = (
            (tip_confidence[indices] >= self.confidence_threshold)
            & np.isfinite(tip_confidence[indices])
            & finite
        )

        frames = self._read_video(episode, indices.tolist())
        proprio = torch.zeros((len(indices), self.max_state_dim), dtype=torch.float32)
        proprio_valid = torch.zeros((len(indices), self.max_state_dim), dtype=torch.bool)
        return {
            "frames": frames,
            "proprio": proprio,
            "proprio_valid": proprio_valid,
            "embodiment_id": 0,
            "fingertip_positions": torch.from_numpy(tip_positions.astype(np.float32, copy=False)),
            "fingertip_valid": torch.from_numpy(valid.astype(bool, copy=False)),
            "dataset_name": f"egodex/{episode.task}",
            "trajectory_id": episode.stem,
            "base_index": int(base_index),
        }

    def __getitem__(self, index: int) -> Dict:
        if self.debug_repeat_batch:
            if not self._debug_cache_initialized:
                repeat_k = int(self.debug_repeat_batch) if isinstance(self.debug_repeat_batch, int) else 1
                repeat_k = max(1, repeat_k)
                self._debug_cache = [self._make_sample(i) for i in range(repeat_k)]
                self._debug_cache_initialized = True
            return self._debug_cache[index % len(self._debug_cache)]
        return self._make_sample(index)
