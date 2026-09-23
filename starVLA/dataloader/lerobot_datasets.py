# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modification: [suport topdowm processing, suport param from config].

import copy
import json
from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import (
    CachedLeRobotSingleDataset,
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
)
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG

def collate_fn(batch):
    return batch


def _cfg_get(data_cfg: Any, key: str, default: Any = None) -> Any:
    if data_cfg is None:
        return default
    if hasattr(data_cfg, "get"):
        return data_cfg.get(key, default)
    return getattr(data_cfg, key, default)


def _should_enable_video_frame_cache(data_cfg: Any, mode: str) -> bool:
    del mode
    if not bool(_cfg_get(data_cfg, "enable_video_frame_cache", False)):
        return False

    legacy_modes = _cfg_get(data_cfg, "video_frame_cache_modes", None)
    if legacy_modes is not None:
        warnings.warn(
            "`video_frame_cache_modes` is deprecated and ignored. "
            "When `enable_video_frame_cache=True`, all dataset modes reuse the shared disk cache.",
            DeprecationWarning,
            stacklevel=2,
        )
    return True


def _use_training_transforms(mode: str) -> bool:
    normalized_mode = str(mode).lower()
    if normalized_mode in {"train", "all"}:
        return True
    if normalized_mode in {"val", "test"}:
        return False
    raise ValueError(f"Unsupported dataset mode `{mode}` for transform selection.")


def _sample_video_delta_indices(action_delta_indices: Sequence[int], num_frames: int) -> list[int]:
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be >= 1, got {num_frames}.")
    action_arr = np.asarray(action_delta_indices, dtype=np.int64).reshape(-1)
    if action_arr.size == 0:
        raise ValueError("`action_delta_indices` cannot be empty when building video sampling indices.")
    if num_frames == 1:
        return [int(action_arr[0])]
    if num_frames == 2:
        return [int(action_arr[0]), int(action_arr[-1])]

    sampled_pos = np.rint(np.linspace(0, action_arr.size - 1, num=num_frames)).astype(np.int64)
    sampled_pos = np.clip(sampled_pos, 0, action_arr.size - 1)
    sampled_pos[0] = 0
    sampled_pos[-1] = action_arr.size - 1
    return action_arr[sampled_pos].astype(np.int64).tolist()


def _as_valid_fps(value: Any) -> float | None:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    return fps if fps > 0 else None


def _resolve_control_fps(
    dataset_path: Path,
    preferred_video_keys: Sequence[str] | None = None,
    default_fps: float = 10.0,
) -> float:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return float(default_fps)

    with open(info_path, "r") as f:
        info = json.load(f)

    dataset_fps = _as_valid_fps(info.get("fps"))
    if dataset_fps is not None:
        return dataset_fps

    features = info.get("features", {})
    action_fps = _as_valid_fps(features.get("action", {}).get("fps"))
    if action_fps is not None:
        return action_fps

    video_features = {
        key: value for key, value in features.items() if isinstance(value, dict) and value.get("dtype") == "video"
    }
    if not video_features:
        print(f"[LeRobotDataset] cannot resolve control fps from {dataset_path}/meta/info.json.")
        print(f"[LeRobotDataset] using default fps: {default_fps}")
        return float(default_fps)

    candidate_feature = None
    if preferred_video_keys:
        for preferred in preferred_video_keys:
            key_candidates = [preferred]
            if preferred.startswith("video."):
                key_candidates.append(preferred.replace("video.", "", 1))
            for key in key_candidates:
                if key in video_features:
                    candidate_feature = video_features[key]
                    break
            if candidate_feature is not None:
                break

    if candidate_feature is None:
        candidate_feature = next(iter(video_features.values()))

    try:
        return float(candidate_feature["video_info"]["video.fps"])
    except (KeyError, TypeError, ValueError):
        pass

    try:
        return float(candidate_feature.get("info", {}).get("video.fps", default_fps))
    except (TypeError, ValueError):
        return float(default_fps)


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    mode: str = "train",
    data_cfg: dict | None = None,
    framework_name: str | None = None,
    dataset_statistics_override: dict[str, Any] | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    if robot_type not in ROBOT_TYPE_CONFIG_MAP:
        available_robot_types = sorted(ROBOT_TYPE_CONFIG_MAP.keys())
        raise ValueError(
            f"Unknown robot_type `{robot_type}`. "
            f"Available robot types in ROBOT_TYPE_CONFIG_MAP: {available_robot_types}."
        )
    data_config = copy.deepcopy(ROBOT_TYPE_CONFIG_MAP[robot_type])
    modality_config = data_config.modality_config()
    is_latent_world = str(framework_name or "").lower() == "lawam"
    image_resolution = int(_cfg_get(data_cfg, "image_resolution", 256))
    transform_image_hw = (image_resolution, image_resolution) if is_latent_world else None
    try:
        transforms = data_config.transform(image_hw=transform_image_hw)
    except TypeError:
        transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        available_robot_types = sorted(ROBOT_TYPE_TO_EMBODIMENT_TAG.keys())
        raise ValueError(
            f"Unknown robot_type `{robot_type}`. "
            f"Available robot types: {available_robot_types}."
        )
    embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    
    raw_num_frames = _cfg_get(data_cfg, "num_frames", 1)
    num_frames = int(raw_num_frames) if raw_num_frames is not None else 1
    sec_chunk = _cfg_get(data_cfg, "sec_chunk", None)
    preferred_video_keys = modality_config.get("video").modality_keys if "video" in modality_config else None
    resolved_action_hz = _resolve_control_fps(
        dataset_path=dataset_path,
        preferred_video_keys=preferred_video_keys,
    )
    if sec_chunk is not None and "action" in modality_config:
        sec_chunk_f = float(sec_chunk)
        if sec_chunk_f <= 0:
            raise ValueError(f"`sec_chunk` must be > 0, got {sec_chunk}.")

        fps = float(resolved_action_hz)
        chunk_len = int(sec_chunk_f * fps)
        if chunk_len < 1:
            raise ValueError(
                f"`sec_chunk` is too small for this dataset fps: sec_chunk={sec_chunk_f}, fps={fps}, "
                f"int(sec_chunk * fps)={chunk_len}. Please increase `sec_chunk`."
            )

        modality_config["action"].delta_indices = list(range(chunk_len))
        print(
            f"[LeRobotDataset] dataset={data_name} robot={robot_type} sec_chunk={sec_chunk_f} "
            f"fps={fps} action_horizon={chunk_len}"
        )

    if "action" in modality_config and "video" in modality_config:
        action_delta_indices = modality_config["action"].delta_indices
        sampled_video_delta = _sample_video_delta_indices(
            action_delta_indices=action_delta_indices,
            num_frames=num_frames,
        )
        modality_config["video"].delta_indices = sampled_video_delta
        print(
            f"[LeRobotDataset] dataset={data_name} robot={robot_type} num_frames={num_frames} "
            f"video_delta_indices={sampled_video_delta}"
        )

    video_backend = _cfg_get(data_cfg, "video_backend", "pyav")
    dataset_cls = CachedLeRobotSingleDataset if _should_enable_video_frame_cache(data_cfg, mode) else LeRobotSingleDataset

    dataset = dataset_cls(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        action_hz=resolved_action_hz,
        mode=mode,
        _val_tail_ratio=float(_cfg_get(data_cfg, "val_tail_ratio", 0.001)),
        video_backend=video_backend,
        data_cfg=data_cfg,
        dataset_statistics_override=dataset_statistics_override,
    )
    if hasattr(dataset, "transforms"):
        if _use_training_transforms(mode):
            dataset.transforms.train()
        else:
            dataset.transforms.eval()
    return dataset

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = True,
    seed: int = 42,
    framework_name: str | None = None,
    dataset_statistics_override: dict[str, Any] | None = None,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix

    data_mix_str = str(data_mix).strip()
    if data_mix_str.startswith("dexwm_robocasa"):
        from starVLA.dataloader.dexwm_robocasa import DexWMRoboCasaDataset

        return DexWMRoboCasaDataset(
            data_root_dir=Path(data_root_dir),
            mode=mode,
            data_cfg=data_cfg,
            seed=seed,
            dataset_statistics_override=dataset_statistics_override,
        )

    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append(
            (
                make_LeRobotSingleDataset(
                    Path(data_root_dir),
                    d_name,
                    robot_type,
                    mode=mode,
                    data_cfg=data_cfg,
                    framework_name=framework_name,
                    dataset_statistics_override=dataset_statistics_override,
                ),
                d_weight,
            )
        )

    random_single_non_wrist_view = bool(
        _cfg_get(
            data_cfg,
            "random_single_non_wrist_view",
            any(
                bool(getattr(ROBOT_TYPE_CONFIG_MAP[robot_type], "random_single_non_wrist_view", False))
                for _, _, robot_type in filtered_mixture_spec
            ),
        )
    )

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        seed=seed,
        random_single_non_wrist_view=random_single_non_wrist_view,
        data_cfg=data_cfg,
        **kwargs,
    )



if __name__ == "__main__":

    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_train_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()
    args.config_yaml = "./starVLA/config/training/starvla_train_oxe.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # cfg.datasets.vla_data.data_mix = "robotwin"
    vla_dataset_cfg = cfg.datasets.vla_data
    # cfg.datasets.vla_data.include_state = True
    vla_dataset_cfg.task_id = 1
    for task_id in ["all"]:
        vla_dataset_cfg.task_id = task_id
        print(f"Testing Task ID: {task_id}")
        dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        # dataset
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm
    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # print(batch)
        # print(1)
        if count > 100:
            break
        count += 1
        pass
