from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from starVLA.model.framework.latent_world.batch_utils import prepare_frame_spatial_uint8

try:
    import h5py
except ImportError as exc:  # pragma: no cover - runtime dependency check
    h5py = None  # type: ignore[assignment]
    _H5PY_IMPORT_ERROR = exc
else:
    _H5PY_IMPORT_ERROR = None


def _cfg_get(data_cfg: Any, key: str, default: Any = None) -> Any:
    if data_cfg is None:
        return default
    if hasattr(data_cfg, "get"):
        return data_cfg.get(key, default)
    return getattr(data_cfg, key, default)


def _cfg_to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    try:
        wrapped_cfg = object.__getattribute__(value, "_cfg")
    except Exception:
        wrapped_cfg = None
    if wrapped_cfg is not None and OmegaConf.is_config(wrapped_cfg):
        return OmegaConf.to_container(wrapped_cfg, resolve=True)
    if isinstance(value, dict):
        return {str(k): _cfg_to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_cfg_to_plain(v) for v in value]
    if not isinstance(value, (str, bytes)):
        if hasattr(value, "items"):
            try:
                return {str(k): _cfg_to_plain(v) for k, v in value.items()}
            except TypeError:
                pass
        try:
            return [_cfg_to_plain(v) for v in value]
        except TypeError:
            pass
    return value


def _as_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(v) for v in value]
    return value


def _stats_from_array(array: np.ndarray) -> dict[str, list[float]]:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array for statistics, got shape {arr.shape}.")
    return {
        "max": np.max(arr, axis=0).astype(np.float64).tolist(),
        "min": np.min(arr, axis=0).astype(np.float64).tolist(),
        "mean": np.mean(arr, axis=0).astype(np.float64).tolist(),
        "std": np.std(arr, axis=0).astype(np.float64).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).astype(np.float64).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).astype(np.float64).tolist(),
    }


def _normalize_text(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _resolve_control_freq(env_args: dict[str, Any] | None, default: float = 1.0) -> float:
    if not isinstance(env_args, dict):
        return float(default)

    env_kwargs = env_args.get("env_kwargs", {})
    if isinstance(env_kwargs, dict):
        control_freq = env_kwargs.get("control_freq", None)
        try:
            if control_freq is not None:
                freq = float(control_freq)
                if freq > 0:
                    return freq
        except (TypeError, ValueError):
            pass

    for key in ("control_freq", "fps"):
        try:
            value = float(env_args.get(key, default))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return float(default)


@dataclass(frozen=True)
class _DexWMEpisode:
    file_path: Path
    demo_name: str
    length: int
    lang: str
    action_hz: float
    subdir: str


@dataclass
class _ArrayStatisticsAccumulator:
    sample_rows_per_episode: int = 256
    count: int = 0
    total: np.ndarray | None = None
    total_sq: np.ndarray | None = None
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    sample_chunks: list[np.ndarray] = field(default_factory=list)

    @staticmethod
    def _validate_2d(array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D array for statistics, got shape {arr.shape}.")
        return arr

    @staticmethod
    def _sample_rows(array: np.ndarray, sample_rows_per_episode: int) -> np.ndarray:
        if sample_rows_per_episode <= 0:
            return np.empty((0, array.shape[-1]), dtype=np.float32)
        if array.shape[0] <= sample_rows_per_episode:
            return np.asarray(array, dtype=np.float32)
        indices = np.linspace(0, array.shape[0] - 1, num=sample_rows_per_episode, dtype=np.int64)
        return np.asarray(array[indices], dtype=np.float32)

    def update(self, array: np.ndarray) -> None:
        arr = self._validate_2d(array)
        if arr.shape[0] == 0:
            return
        if self.total is None:
            dim = int(arr.shape[1])
            self.total = np.zeros(dim, dtype=np.float64)
            self.total_sq = np.zeros(dim, dtype=np.float64)
            self.minimum = np.full(dim, np.inf, dtype=np.float64)
            self.maximum = np.full(dim, -np.inf, dtype=np.float64)
        elif arr.shape[1] != int(self.total.shape[0]):
            raise ValueError(
                "Statistics accumulator received arrays with mismatched feature dimensions: "
                f"got {arr.shape[1]}, expected {self.total.shape[0]}."
            )

        self.count += int(arr.shape[0])
        self.total += arr.sum(axis=0)
        self.total_sq += np.square(arr).sum(axis=0)
        self.minimum = np.minimum(self.minimum, np.min(arr, axis=0))
        self.maximum = np.maximum(self.maximum, np.max(arr, axis=0))
        sample = self._sample_rows(arr, int(self.sample_rows_per_episode))
        if sample.shape[0] > 0:
            self.sample_chunks.append(sample)

    def finalize(self) -> dict[str, list[float]]:
        if self.count <= 0 or self.total is None or self.total_sq is None:
            raise ValueError("Cannot finalize empty statistics accumulator.")
        assert self.minimum is not None
        assert self.maximum is not None
        mean = self.total / float(self.count)
        var = self.total_sq / float(self.count) - np.square(mean)
        var = np.clip(var, 0.0, None)
        std = np.sqrt(var)
        if self.sample_chunks:
            samples = np.concatenate(self.sample_chunks, axis=0)
            q01 = np.quantile(samples, 0.01, axis=0)
            q99 = np.quantile(samples, 0.99, axis=0)
        else:
            q01 = self.minimum
            q99 = self.maximum
        return {
            "max": self.maximum.astype(np.float64).tolist(),
            "min": self.minimum.astype(np.float64).tolist(),
            "mean": mean.astype(np.float64).tolist(),
            "std": std.astype(np.float64).tolist(),
            "q01": np.asarray(q01, dtype=np.float64).tolist(),
            "q99": np.asarray(q99, dtype=np.float64).tolist(),
        }


class DexWMRoboCasaDataset(Dataset):
    """
    Minimal raw HDF5 adapter for DexWM RoboCasa data.

    It emits LaWAM raw samples directly, without converting the source data to
    LeRobot V2.1 / parquet format first.
    """

    _STATE_DIM = 25
    _ACTION_DIM = 25
    _EMBODIMENT_ID = int(EMBODIMENT_TAG_MAPPING[EmbodimentTag.NEW_EMBODIMENT.value])
    _DEFAULT_PRIMARY_KEYS = ("robot0_robotview_2_image", "robot0_robotview_image")
    _DEFAULT_WRIST_KEY = "gripper0_right_right_eye_in_hand_image"
    _DEFAULT_POSE_KEYS = (
        "robot0_right_eef_T_right_base_pos",
        "robot0_right_eef_pos",
        "robot0_right_eef_T_world_pos",
    )
    _DEFAULT_QUAT_KEYS = (
        "robot0_right_eef_T_right_base_quat_xyzw",
        "robot0_right_eef_quat",
        "robot0_right_eef_T_world_quat_xyzw",
    )
    _DEFAULT_ACTION_KEYS = (
        "abs_right_arm_base_action",
        "abs_action",
        "actions",
    )

    def __init__(
        self,
        *,
        data_root_dir: Path,
        mode: str = "train",
        data_cfg: Any | None = None,
        seed: int = 42,
        dataset_statistics_override: dict[str, Any] | None = None,
    ) -> None:
        if h5py is None:  # pragma: no cover - runtime dependency check
            raise ImportError(
                "DexWMRoboCasaDataset requires `h5py`. Install it in the training environment "
                "before using the DexWM data path."
            ) from _H5PY_IMPORT_ERROR

        self.data_root_dir = Path(data_root_dir).expanduser()
        self.mode = str(mode).lower()
        if self.mode not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unsupported dataset mode `{mode}` for DexWM RoboCasa.")

        self.seed = int(seed)
        self.data_cfg = data_cfg
        self.dataset_statistics_override = dataset_statistics_override
        self.image_resolution = int(_cfg_get(data_cfg, "image_resolution", 256))
        self.num_frames = max(1, int(_cfg_get(data_cfg, "num_frames", 2)))
        self.val_tail_ratio = float(_cfg_get(data_cfg, "val_tail_ratio", 0.01))
        self.action_horizon = int(_cfg_get(data_cfg, "action_horizon", 4))
        self.action_horizon = max(1, self.action_horizon)
        action_hz_override = _cfg_get(data_cfg, "action_hz_override", None)
        self.action_hz_override = None if action_hz_override is None else float(action_hz_override)
        self.stats_sample_rows_per_episode = max(1, int(_cfg_get(data_cfg, "stats_sample_rows_per_episode", 256)))
        self.primary_video_keys = tuple(
            str(key) for key in _cfg_get(data_cfg, "primary_video_keys", self._DEFAULT_PRIMARY_KEYS)
        )
        if not self.primary_video_keys:
            raise ValueError("`datasets.vla_data.primary_video_keys` must contain at least one view key.")
        self.wm_primary_video_key = str(_cfg_get(data_cfg, "wm_primary_video_key", self.primary_video_keys[0]))
        self.wrist_video_key = str(_cfg_get(data_cfg, "wrist_video_key", self._DEFAULT_WRIST_KEY))
        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.embodiment_id = self._EMBODIMENT_ID
        self._split_file_names = self._resolve_split_file_names()

        self._episodes = self._scan_episodes()
        self._samples = self._build_sample_index()
        self._file_cache: dict[Path, h5py.File] = {}
        self._epoch = 0
        self._statistics_cache: dict[str, Any] | None = None
        self._state_stats: dict[str, list[float]] | None = None
        self._action_stats: dict[str, list[float]] | None = None
        self._init_normalization_stats()

    def __len__(self) -> int:
        return len(self._samples)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def close(self) -> None:
        for file_handle in self._file_cache.values():
            try:
                file_handle.close()
            except Exception:
                pass
        self._file_cache.clear()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _resolve_subdirs(self) -> list[str]:
        mix = str(_cfg_get(self.data_cfg, "data_mix", "dexwm_robocasa_random")).strip().lower()
        if mix in {"dexwm_robocasa_random", "dexwm_robocasa", "random"}:
            return ["exploratory_movements", "exploratory_movement", "gripper_open_and_close"]
        if mix in {"dexwm_robocasa_pick_place", "pick_place", "pick-and-place-2.0"}:
            return ["pick-and-place-2.0"]
        if mix in {"dexwm_robocasa_all", "all"}:
            return ["exploratory_movements", "exploratory_movement", "gripper_open_and_close", "pick-and-place-2.0"]
        return ["exploratory_movements", "exploratory_movement", "gripper_open_and_close"]

    @staticmethod
    def _normalize_file_name(value: Any) -> str:
        if isinstance(value, int):
            return f"combine_demos_{value}.hdf5"
        text = str(value).strip()
        if text.isdigit():
            return f"combine_demos_{text}.hdf5"
        return text

    def _resolve_split_file_names(self) -> set[str] | None:
        split_files = _cfg_to_plain(_cfg_get(self.data_cfg, "split_files", None))
        mode_key = self.mode

        values: Any = None
        if isinstance(split_files, dict):
            values = split_files.get(mode_key)
            if values is None and self.mode == "all":
                values = split_files.get("all")
        if values is None:
            values = _cfg_to_plain(_cfg_get(self.data_cfg, f"{mode_key}_files", None))
        if values is None and self.mode == "test":
            values = _cfg_to_plain(_cfg_get(self.data_cfg, "test_files", None))
        if values is None:
            return None
        if isinstance(values, (str, int)):
            values = [values]
        return {self._normalize_file_name(value) for value in values}

    @staticmethod
    def _load_json_attr(attrs: Any, key: str) -> dict[str, Any] | None:
        if key not in attrs:
            return None
        raw = attrs[key]
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return json.loads(raw)
        except Exception:
            return None

    def _scan_episodes(self) -> list[_DexWMEpisode]:
        episodes: list[_DexWMEpisode] = []
        subdirs = self._resolve_subdirs()
        for subdir in subdirs:
            root = self.data_root_dir / subdir
            if not root.exists():
                continue
            for file_path in sorted(root.glob("*.hdf5")):
                if self._split_file_names is not None and file_path.name not in self._split_file_names:
                    continue
                with h5py.File(file_path, "r") as f:
                    if "data" not in f:
                        continue
                    env_args = self._load_json_attr(f["data"].attrs, "env_args")
                    action_hz = _resolve_control_freq(env_args, default=1.0)
                    for demo_name in sorted(f["data"].keys()):
                        demo_group = f["data"][demo_name]
                        obs = demo_group.get("obs", None)
                        if obs is None:
                            continue
                        if not self._demo_has_required_keys(obs):
                            continue
                        length = int(demo_group["actions"].shape[0]) if "actions" in demo_group else int(
                            obs[self._DEFAULT_PRIMARY_KEYS[0]].shape[0]
                        )
                        if length <= 0:
                            continue
                        lang = self._resolve_language(demo_group, fallback=subdir)
                        episodes.append(
                            _DexWMEpisode(
                                file_path=file_path,
                                demo_name=str(demo_name),
                                length=length,
                                lang=lang,
                                action_hz=float(
                                    self.action_hz_override if self.action_hz_override is not None else action_hz
                                ),
                                subdir=subdir,
                            )
            )
        if not episodes:
            raise FileNotFoundError(
                f"No DexWM RoboCasa episodes found under `{self.data_root_dir}` for split `{self.mode}` "
                f"and files `{sorted(self._split_file_names) if self._split_file_names is not None else 'ALL'}`."
            )
        return episodes

    def _build_sample_index(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for episode_id, episode in enumerate(self._episodes):
            usable_length = max(1, episode.length - self.action_horizon + 1)
            if self._split_file_names is not None:
                start_range = range(0, usable_length)
            else:
                split_point = int(round((1.0 - self.val_tail_ratio) * usable_length))
                split_point = min(max(split_point, 1), usable_length)
                if self.mode == "train":
                    start_range = range(0, split_point)
                elif self.mode in {"val", "test"}:
                    start_range = range(split_point, usable_length)
                else:
                    start_range = range(0, usable_length)

            for start_idx in start_range:
                samples.append(
                    {
                        "episode_id": episode_id,
                        "start_idx": int(start_idx),
                    }
                )

        if self.mode in {"train", "all"}:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(samples)
        return samples

    @staticmethod
    def _resolve_language(demo_group: Any, *, fallback: str) -> str:
        ep_meta = demo_group.attrs.get("ep_meta", None)
        if ep_meta is not None:
            try:
                if isinstance(ep_meta, bytes):
                    ep_meta = ep_meta.decode("utf-8")
                meta = json.loads(ep_meta)
                lang = meta.get("free_form_lang") or meta.get("lang")
                if lang:
                    return str(lang)
            except Exception:
                pass
        fallback_map = {
            "exploratory_movements": "dexwm exploratory hand movement",
            "gripper_open_and_close": "dexwm gripper open and close",
            "pick-and-place-2.0": "dexwm pick and place",
        }
        return fallback_map.get(fallback, f"dexwm {fallback.replace('_', ' ')}")

    def _demo_has_required_keys(self, obs_group: Any) -> bool:
        required = {
            "robot0_right_gripper_qpos",
            "robot0_right_gripper_keypoint_pose",
            "robot0_right_hand_T_world_pose_mat",
        }
        if not required.issubset(set(obs_group.keys())):
            return False
        if not any(key in obs_group for key in DexWMRoboCasaDataset._DEFAULT_POSE_KEYS):
            return False
        if not any(key in obs_group for key in DexWMRoboCasaDataset._DEFAULT_QUAT_KEYS):
            return False
        if not all(key in obs_group for key in self.primary_video_keys):
            return False
        if self.wm_primary_video_key not in obs_group:
            return False
        if self.wrist_video_key not in obs_group:
            return False
        return True

    def _get_file(self, file_path: Path) -> h5py.File:
        file_path = Path(file_path)
        cached = self._file_cache.get(file_path)
        if cached is not None:
            return cached
        handle = h5py.File(file_path, "r")
        self._file_cache[file_path] = handle
        return handle

    def _episode(self, episode_id: int) -> _DexWMEpisode:
        return self._episodes[int(episode_id)]

    def _init_normalization_stats(self) -> None:
        stats = self.dataset_statistics_override or self.build_dataset_statistics()
        tag_stats = stats[self.tag]
        self._state_stats = tag_stats["state"]
        self._action_stats = tag_stats["action"]

    @staticmethod
    def _normalize_array(array: np.ndarray, stats: dict[str, list[float]]) -> np.ndarray:
        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        high_key = "max" if "max" in stats else "q99"
        low_key = "min" if "min" in stats else "q01"
        high = np.asarray(stats[high_key], dtype=np.float32)
        low = np.asarray(stats[low_key], dtype=np.float32)
        if arr.shape[-1] != high.shape[0]:
            raise ValueError(f"Dim mismatch when normalizing: got {arr.shape[-1]}, expected {high.shape[0]}.")
        denom = high - low
        out = arr.copy()
        valid = np.abs(denom) > 1e-12
        if np.any(valid):
            out[..., valid] = (out[..., valid] - low[valid]) / denom[valid] * 2.0 - 1.0
        if np.any(~valid):
            out[..., ~valid] = 0.0
        return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _sample_indices(start_idx: int, end_idx: int, num_frames: int) -> list[int]:
        if num_frames <= 1:
            return [int(start_idx)]
        if end_idx <= start_idx:
            return [int(start_idx)] * int(num_frames)
        if num_frames == 2:
            return [int(start_idx), int(end_idx)]
        lin = np.linspace(start_idx, end_idx, num=num_frames)
        return np.rint(lin).astype(np.int64).tolist()

    def _load_view_frames(
        self,
        obs_group: Any,
        key: str,
        frame_indices: Iterable[int],
    ) -> torch.Tensor:
        if key not in obs_group:
            raise KeyError(f"Missing observation key `{key}` in DexWM episode.")
        frames = []
        raw = obs_group[key]
        length = int(raw.shape[0])
        for idx in frame_indices:
            clipped_idx = int(np.clip(int(idx), 0, max(0, length - 1)))
            frame = np.asarray(raw[clipped_idx])
            frame = prepare_frame_spatial_uint8(
                frame,
                target_hw=(self.image_resolution, self.image_resolution),
            )
            frames.append(frame)
        return torch.stack(frames, dim=0)

    def _load_proprio(
        self,
        obs_group: Any,
        indices: Iterable[int],
    ) -> np.ndarray:
        pos_key = next((key for key in self._DEFAULT_POSE_KEYS if key in obs_group), None)
        quat_key = next((key for key in self._DEFAULT_QUAT_KEYS if key in obs_group), None)
        if pos_key is None or quat_key is None:
            raise KeyError("DexWM episode is missing end-effector pose keys.")
        gripper_key = "robot0_right_gripper_qpos"
        if gripper_key not in obs_group:
            raise KeyError("DexWM episode is missing `robot0_right_gripper_qpos`.")

        idx_list = [int(i) for i in indices]
        pos = np.asarray(obs_group[pos_key][idx_list], dtype=np.float32)
        quat = np.asarray(obs_group[quat_key][idx_list], dtype=np.float32)
        if quat.shape[-1] != 4:
            raise ValueError(f"Expected quaternion shape [T,4] for `{quat_key}`, got {quat.shape}.")
        rot6 = R.from_quat(quat).as_matrix()[:, :2, :].reshape(len(idx_list), 6).astype(np.float32)
        gripper = np.asarray(obs_group[gripper_key][idx_list], dtype=np.float32)
        proprio = np.concatenate([pos, rot6, gripper], axis=-1)
        if proprio.shape[-1] != self._STATE_DIM:
            raise ValueError(
                f"DexWM proprio dim mismatch: expected {self._STATE_DIM}, got {proprio.shape[-1]}."
            )
        return proprio.astype(np.float32, copy=False)

    @staticmethod
    def _axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
        axis_angle = np.asarray(axis_angle, dtype=np.float32)
        if axis_angle.ndim != 2 or axis_angle.shape[-1] != 3:
            raise ValueError(f"Expected axis-angle array with shape [T,3], got {axis_angle.shape}.")
        rot_mats = R.from_rotvec(axis_angle).as_matrix()
        return rot_mats[:, :2, :].reshape(axis_angle.shape[0], 6).astype(np.float32)

    def _load_action(self, demo_group: Any, indices: Iterable[int]) -> np.ndarray:
        action_source = None
        for key in self._DEFAULT_ACTION_KEYS:
            if key in demo_group:
                action_source = key
                break
        if action_source is None:
            raise KeyError(
                "DexWM episode is missing action arrays. Expected one of "
                f"{self._DEFAULT_ACTION_KEYS}."
            )

        raw = np.asarray(demo_group[action_source], dtype=np.float32)
        idx_list = [int(i) for i in indices]
        rows = raw[idx_list]
        if rows.ndim != 2 or rows.shape[-1] < 31:
            raise ValueError(
                f"Unexpected action shape for `{action_source}`: got {rows.shape}, expected [T, >=31]."
            )

        pos = rows[:, :3]
        rot6 = self._axis_angle_to_rot6d(rows[:, 3:6])
        gripper = rows[:, 15:31].astype(np.float32, copy=False)
        action = np.concatenate([pos, rot6, gripper], axis=-1)
        if action.shape[-1] != self._ACTION_DIM:
            raise ValueError(
                f"DexWM action dim mismatch: expected {self._ACTION_DIM}, got {action.shape[-1]}."
            )
        return action.astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[int(index)]
        episode = self._episode(sample["episode_id"])
        file_handle = self._get_file(episode.file_path)
        demo_group = file_handle["data"][episode.demo_name]
        obs_group = demo_group["obs"]

        length = int(episode.length)
        start_idx = int(sample["start_idx"])
        action_end_idx = min(start_idx + self.action_horizon - 1, length - 1)
        action_indices = [min(start_idx + offset, length - 1) for offset in range(self.action_horizon)]
        primary_frame_indices = self._sample_indices(start_idx, action_end_idx, self.num_frames)

        primary_views = []
        primary_views_by_key = {}
        for key in self.primary_video_keys:
            if key not in obs_group:
                continue
            view_frames = self._load_view_frames(obs_group, key, primary_frame_indices)
            primary_views.append(view_frames)
            primary_views_by_key[key] = view_frames
        if not primary_views:
            raise KeyError(
                f"Episode `{episode.demo_name}` does not contain any configured primary view keys: "
                f"{self.primary_video_keys}"
            )
        if self.wm_primary_video_key in primary_views_by_key:
            wm_primary_video = primary_views_by_key[self.wm_primary_video_key]
        else:
            wm_primary_video = self._load_view_frames(obs_group, self.wm_primary_video_key, primary_frame_indices)

        wrist_frames = self._load_view_frames(obs_group, self.wrist_video_key, [start_idx])
        state = self._load_proprio(obs_group, [start_idx])
        action = self._load_action(demo_group, action_indices)

        if self._state_stats is None or self._action_stats is None:
            raise RuntimeError("DexWM normalization statistics were not initialized.")
        state = self._normalize_array(state, self._state_stats)
        action = self._normalize_array(action, self._action_stats)

        return {
            "primary_videos": torch.stack(primary_views, dim=0).to(dtype=torch.uint8),
            "wm_primary_video": wm_primary_video.to(dtype=torch.uint8),
            "wrist_images": wrist_frames.to(dtype=torch.uint8),
            "lang": episode.lang,
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "embodiment_id": int(self.embodiment_id),
            "action_hz": float(episode.action_hz),
        }

    def build_dataset_statistics(self) -> dict[str, Any]:
        if self._statistics_cache is not None:
            return self._statistics_cache

        state_acc = _ArrayStatisticsAccumulator(sample_rows_per_episode=self.stats_sample_rows_per_episode)
        action_acc = _ArrayStatisticsAccumulator(sample_rows_per_episode=self.stats_sample_rows_per_episode)
        for episode in self._episodes:
            file_handle = self._get_file(episode.file_path)
            demo_group = file_handle["data"][episode.demo_name]
            obs_group = demo_group["obs"]
            frame_indices = list(range(int(episode.length)))
            proprio = self._load_proprio(obs_group, frame_indices)
            action = self._load_action(demo_group, frame_indices)
            state_acc.update(proprio)
            action_acc.update(action)

        stats = {
            self.tag: {
                "state": state_acc.finalize(),
                "action": action_acc.finalize(),
                "num_transitions": int(len(self._samples)),
                "num_trajectories": int(len(self._episodes)),
            }
        }
        self._statistics_cache = _as_json(stats)
        return self._statistics_cache
