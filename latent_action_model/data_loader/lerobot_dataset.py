import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.data_config_lam import (
    build_lam_dataset_transform,
    filter_lam_video_keys,
)
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    ModalityConfig,
)
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import (
    EmbodimentTag,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
)


def _build_modality_config(
    robot_type: str,
    num_frames: int,
    preferred_video_key: Optional[str] = None,
    state_keys: Optional[Sequence[str]] = None,
    include_action: bool = False,
    include_language: bool = False,
) -> Dict[str, ModalityConfig]:
    """
    Clone ROBOT_TYPE_CONFIG_MAP[robot_type].modality_config() and override delta_indices.
    We only keep video/state by default to reduce IO for LAM.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    base_cfg = ROBOT_TYPE_CONFIG_MAP[robot_type].modality_config()
    default_delta = list(range(num_frames))

    def override_delta(cfg: ModalityConfig, keys: Optional[Sequence[str]]) -> ModalityConfig:
        new_keys = list(keys) if keys is not None else cfg.modality_keys
        return ModalityConfig(delta_indices=list(default_delta), modality_keys=new_keys)

    # video
    video_keys = filter_lam_video_keys(
        base_video_keys=base_cfg["video"].modality_keys,
        preferred_video_key=preferred_video_key,
    )
    video_modality = override_delta(base_cfg["video"], video_keys)

    # state
    state_modality = override_delta(base_cfg["state"], state_keys)

    modality_configs: Dict[str, ModalityConfig] = {
        "video": video_modality,
        "state": state_modality,
    }

    if include_action and "action" in base_cfg:
        modality_configs["action"] = override_delta(base_cfg["action"], base_cfg["action"].modality_keys)
    if include_language and "language" in base_cfg:
        modality_configs["language"] = base_cfg["language"]

    return modality_configs


def _build_temporal_delta_indices(num_frames: int, frame_dt_sec: float, fps: float) -> np.ndarray:
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if frame_dt_sec <= 0:
        raise ValueError(f"frame_dt_sec must be > 0, got {frame_dt_sec}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    stride = max(1, int(round(frame_dt_sec * fps)))
    return np.arange(0, num_frames * stride, stride, dtype=np.int64)


class LeRobotLAMDataset(Dataset):
    """
    Mixture dataset that reuses starVLA's LeRobot loaders but emits LAM-ready raw samples.

    Output sample:
        {
            "frames": torch.Tensor[T, C, H, W] uint8,
            "proprio": torch.Tensor[T, D] float32,
            "embodiment_id": int,
        }
    """

    def __init__(
        self,
        data_root_dir: str | Path,
        data_mix: str,
        num_frames: int,
        mode: str = "train",
        val_tail_ratio: float = 0.01,
        video_backend: str = "pyav",
        preferred_video_key: Optional[str] = None,
        state_keys: Optional[Sequence[str]] = None,
        image_hw: Tuple[int, int] = (256, 256),
        *,
        frame_dt_sec: float,
        human_frame_dt_sec: Optional[float] = None,
        fingertip_sidecar_root: Optional[str | Path] = None,
        max_retries: int = 5,
        debug_repeat_batch: Union[bool, int] = False,
    ) -> None:
        super().__init__()
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        if frame_dt_sec <= 0:
            raise ValueError(f"frame_dt_sec must be > 0, got {frame_dt_sec}")
        if human_frame_dt_sec is not None and human_frame_dt_sec <= 0:
            raise ValueError(
                f"human_frame_dt_sec must be > 0 when provided, got {human_frame_dt_sec}"
            )
        normalized_mode = str(mode).lower()
        if normalized_mode not in {"train", "val", "test", "all"}:
            raise ValueError(
                f"Unsupported mode `{mode}`. Expected one of ['train', 'val', 'test', 'all']."
            )
        if not (0.0 <= float(val_tail_ratio) < 1.0):
            raise ValueError(
                f"val_tail_ratio must be in [0.0, 1.0), got {val_tail_ratio}"
            )

        self.data_root_dir = Path(data_root_dir)
        self.data_mix = data_mix
        self.num_frames = num_frames
        self.mode = normalized_mode
        self.val_tail_ratio = float(val_tail_ratio)
        self.preferred_video_key = preferred_video_key
        self.state_keys = list(state_keys) if state_keys else None
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.frame_dt_sec = frame_dt_sec
        self.human_frame_dt_sec = human_frame_dt_sec
        self.fingertip_sidecar_root = (
            Path(fingertip_sidecar_root).expanduser()
            if fingertip_sidecar_root is not None
            else None
        )
        self.max_retries = max_retries


        self.video_backend = video_backend
        
        # Debug mode: cache and repeat samples
        self.debug_repeat_batch = debug_repeat_batch
        self._cached_samples: List[Dict] = []
        self._cache_initialized: bool = False

        data_cfg = {
            "data_root_dir": str(self.data_root_dir),
            "video_backend": self.video_backend,
            "fingertip_sidecar_root": (
                str(self.fingertip_sidecar_root)
                if self.fingertip_sidecar_root is not None
                else None
            ),
        }

        mixture_spec = DATASET_NAMED_MIXTURES[self.data_mix]
        # dedupe (dataset_name, robot_type)
        seen: set[Tuple[str, str]] = set()
        dataset_mixture: List[Tuple[LeRobotSingleDataset, float]] = []

        for dataset_name, weight, robot_type in mixture_spec:
            key = (dataset_name, robot_type)
            if key in seen:
                continue
            seen.add(key)

            modality_cfg = _build_modality_config(
                robot_type=robot_type,
                num_frames=self.num_frames,
                preferred_video_key=self.preferred_video_key,
                state_keys=self.state_keys,
                include_action=False,
                include_language=False,
            )
            # Map robot_type to EmbodimentTag
            embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG.get(robot_type)
            if embodiment_tag is None:
                raise ValueError(f"Robot type '{robot_type}' not found in ROBOT_TYPE_TO_EMBODIMENT_TAG mapping")

            ds = LeRobotSingleDataset(
                dataset_path=self.data_root_dir / dataset_name,
                modality_configs=modality_cfg,
                embodiment_tag=embodiment_tag,
                mode=self.mode,
                _val_tail_ratio=self.val_tail_ratio,
                video_backend=self.video_backend,
                data_cfg=data_cfg,
            )
            self._apply_temporal_delta_indices(ds)
            ds._lam_transforms_by_video_key = self._build_video_key_transform_cache(
                dataset=ds,
                robot_type=robot_type,
                state_keys=modality_cfg["state"].modality_keys,
                video_keys=modality_cfg["video"].modality_keys,
            )
            dataset_mixture.append((ds, weight))

        self.mixture = LeRobotMixtureDataset(
            dataset_mixture,
            mode=self.mode,
            balance_dataset_weights=True,
            seed=42,
            random_single_non_wrist_view=True,
            data_cfg=data_cfg,
        )

    def _apply_temporal_delta_indices(self, dataset: LeRobotSingleDataset) -> None:
        video_keys = dataset.modality_keys.get("video", [])
        if not video_keys:
            raise RuntimeError(f"Dataset {dataset.dataset_name} has no video keys configured.")

        fps = float(dataset.action_hz)
        is_human = dataset.tag == EmbodimentTag.HUMAN.value
        effective_dt_sec = (
            self.human_frame_dt_sec
            if is_human and self.human_frame_dt_sec is not None
            else self.frame_dt_sec
        )
        delta_arr = _build_temporal_delta_indices(
            num_frames=self.num_frames,
            frame_dt_sec=effective_dt_sec,
            fps=fps,
        )
        stride = int(delta_arr[1] - delta_arr[0]) if len(delta_arr) > 1 else 1
        print(
            f"[LeRobotLAMDataset][temporal] dataset={dataset.dataset_name} "
            f"tag={dataset.tag} fps={fps:.3f} dt_sec={effective_dt_sec:.6f} stride={stride}"
        )

        for k in video_keys:
            dataset._delta_indices[k] = delta_arr.copy()
        for k in dataset.modality_keys.get("state", []):
            dataset._delta_indices[k] = delta_arr.copy()

    def __len__(self) -> int:
        # In debug mode, return a large virtual length to support multiple epochs
        if self.debug_repeat_batch:
            return 10000
        return len(self.mixture)

    def _build_video_key_transform_cache(
        self,
        dataset: LeRobotSingleDataset,
        robot_type: str,
        state_keys: Sequence[str],
        video_keys: Sequence[str],
    ) -> Dict[str, ComposedModalityTransform]:
        transforms_by_video_key: Dict[str, ComposedModalityTransform] = {}
        for video_key in video_keys:
            lam_transform = build_lam_dataset_transform(
                robot_type=robot_type,
                image_hw=self.image_hw,
                state_keys=state_keys,
                video_keys=[video_key],
            )
            lam_transform.set_metadata(dataset.metadata)
            if self.mode == "train":
                lam_transform.train()
            else:
                lam_transform.eval()
            transforms_by_video_key[video_key] = lam_transform
        return transforms_by_video_key

    def _try_get_sample(self, index: int) -> Dict:
        dataset, traj_id, base_index, selected_video_keys = self.mixture._sample_candidate(
            index,
            max_video_retries=self.max_retries,
        )
        raw_data = dataset.get_step_data(
            traj_id,
            base_index,
            modality_keys_override={"video": selected_video_keys},
        )

        # video: only use non-wrist keys selected by mixture sampling.
        available_video_keys = [k for k in selected_video_keys if "wrist" not in k.lower()]
        if not available_video_keys:
            raise RuntimeError(
                f"No non-wrist video key selected for dataset {dataset.dataset_name}. "
                f"selected_video_keys={selected_video_keys}"
            )
        if self.preferred_video_key and self.preferred_video_key in available_video_keys:
            video_key = self.preferred_video_key
        else:
            video_key = available_video_keys[0]
        lam_transform = dataset._lam_transforms_by_video_key[video_key]
        data = lam_transform(raw_data)
        frames = data[video_key]
        if isinstance(frames, torch.Tensor):
            frames_t = frames.contiguous()
            if frames_t.dtype != torch.uint8:
                frames_t = frames_t.to(torch.uint8)
        else:
            frames_t = torch.from_numpy(np.ascontiguousarray(frames)).to(torch.uint8)
        if frames_t.ndim != 4 or frames_t.shape[1] != 3:
            raise ValueError(
                f"LAM dataset expects frames as [T, C, H, W], got {tuple(frames_t.shape)} "
                f"for dataset={dataset.dataset_name} video_key={video_key}."
            )

        # state (proprio)
        proprio_tensors: List[torch.Tensor] = []
        for state_key in dataset.modality_keys.get("state", []):
            state_value = data[state_key]
            if isinstance(state_value, torch.Tensor):
                state_tensor = state_value
            else:
                state_tensor = torch.from_numpy(np.asarray(state_value))
            proprio_tensors.append(state_tensor.to(torch.float32))

        if not proprio_tensors:
            raise RuntimeError("No state keys found for LAM dataset.")
        proprio = torch.cat(proprio_tensors, dim=-1).contiguous()

        fingertip_data = dataset.get_fingertip_data(traj_id, base_index)
        fingertip_positions = None
        fingertip_valid = None
        if fingertip_data is not None:
            fingertip_positions, fingertip_valid = fingertip_data
            fingertip_positions = torch.from_numpy(
                np.ascontiguousarray(fingertip_positions)
            ).to(torch.float32)
            fingertip_valid = torch.from_numpy(
                np.ascontiguousarray(fingertip_valid)
            ).to(torch.bool)

        embodiment_id = int(dataset.embodiment_id)

        sample = {
            "frames": frames_t,
            "proprio": proprio,
            "embodiment_id": embodiment_id,
            # Keep sample provenance for debugging late-step loss spikes.
            "dataset_name": str(dataset.dataset_name),
            "trajectory_id": int(traj_id) if isinstance(traj_id, (int, np.integer)) else str(traj_id),
            "base_index": int(base_index),
        }
        if fingertip_positions is not None:
            sample["fingertip_positions"] = fingertip_positions
            sample["fingertip_valid"] = fingertip_valid
        return sample

    def __getitem__(self, index: int) -> Dict:
        # === Debug mode: cache and repeat samples ===
        if self.debug_repeat_batch:
            if not self._cache_initialized:
                # Initialize cache with k samples
                repeat_k = int(self.debug_repeat_batch) if isinstance(self.debug_repeat_batch, int) else 1
                repeat_k = max(1, repeat_k)
                print(f"[LeRobotLAMDataset] Debug mode: Caching {repeat_k} sample(s) for repeated use...")
                
                for i in range(repeat_k):
                    last_err: Optional[Exception] = None
                    for _ in range(self.max_retries):
                        try:
                            sample = self._try_get_sample(i)
                            self._cached_samples.append(sample)
                            break
                        except Exception as e:
                            last_err = e
                            continue
                    if last_err:
                        raise RuntimeError(f"Failed to cache sample {i} after retries: {last_err}")
                
                self._cache_initialized = True
                print(f"[LeRobotLAMDataset] Debug mode: Successfully cached {len(self._cached_samples)} sample(s).")
            
            # Return samples from cache in a round-robin manner
            cache_idx = index % len(self._cached_samples)
            return self._cached_samples[cache_idx]
        
        # === Normal mode ===
        last_err: Optional[Exception] = None
        for _ in range(self.max_retries):
            try:
                return self._try_get_sample(index)
            except Exception as e:  # retry on IO/video issues
                last_err = e
                index = random.randint(0, len(self.mixture) - 1)
                continue
        if last_err:
            raise last_err
        raise RuntimeError("Failed to fetch sample after retries.")
