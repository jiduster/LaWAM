import copy
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader

logger = get_logger(__name__)


def _cfg_get(data_cfg, key, default=None):
    if data_cfg is None:
        return default
    if hasattr(data_cfg, "get"):
        return data_cfg.get(key, default)
    return getattr(data_cfg, key, default)


def _vla_worker_init_fn(_worker_id: int) -> None:
    """Cap per-worker CPU thread counts to reduce dataloader tail latency."""
    raw_threads = os.environ.get("STARVLA_WORKER_OMP_THREADS", "1")
    os.environ["OMP_NUM_THREADS"] = raw_threads
    os.environ["MKL_NUM_THREADS"] = raw_threads
    os.environ["OPENBLAS_NUM_THREADS"] = raw_threads
    os.environ["NUMEXPR_NUM_THREADS"] = raw_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = raw_threads
    os.environ["BLIS_NUM_THREADS"] = raw_threads

    threads = int(raw_threads) if raw_threads.isdigit() else 1
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(threads, 2)))
    except RuntimeError:
        pass


def _build_latent_world_collator(cfg, *, policy_cfg, training: bool):
    from starVLA.dataloader.latent_world_train_collator import LatentWorldTrainCollator
    from starVLA.model.framework.latent_world.processor_utils import build_latent_world_processor_spec

    vla_dataset_cfg = cfg.datasets.vla_data
    processor_spec = build_latent_world_processor_spec(
        policy_cfg=policy_cfg,
        vlm_model_id=str(cfg.framework.qwenvl.base_vlm),
    )
    collator = LatentWorldTrainCollator(
        policy_cfg=policy_cfg,
        processor_spec=processor_spec,
        act_queries=int(policy_cfg.num_action_queries),
        flow_queries=int(policy_cfg.flow_action_num_queries),
        enable_primary_video_aug=bool(vla_dataset_cfg.get("enable_primary_video_aug", False)),
        enable_primary_random_resized_crop=bool(vla_dataset_cfg.get("enable_primary_random_resized_crop", False)),
        cot_prompt_before_wrist=vla_dataset_cfg.get("CoT_prompt_before_wrist", None),
        cot_prompt_after_wrist=vla_dataset_cfg.get("CoT_prompt_after_wrist", None),
    )
    return collator.train() if training else collator.eval()

def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    return value


def save_dataset_statistics(dataset_statistics, output_path):
    """Save dataset statistics to JSON."""
    out_path = Path(output_path)
    if out_path.suffix != ".json":
        out_path = out_path / "dataset_statistics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_statistics = _to_jsonable(copy.deepcopy(dataset_statistics))
    with open(out_path, "w", encoding="utf-8") as f_json:
        json.dump(serializable_statistics, f_json, indent=2, ensure_ascii=False)
    logger.info("Saved dataset statistics file at path %s", out_path)


def _load_pretrained_dataset_statistics(pretrained_checkpoint: str | Path) -> tuple[Optional[dict], Optional[Path]]:
    checkpoint_path = Path(str(pretrained_checkpoint)).expanduser()
    try:
        run_dir = checkpoint_path.parents[1]
    except IndexError:
        logger.warning(
            "Unable to resolve pretrained run directory from checkpoint path `%s`; fallback to current dataset statistics.",
            checkpoint_path,
        )
        return None, None

    stats_path = run_dir / "dataset_statistics.json"
    if not stats_path.exists():
        logger.warning(
            "Pretrained dataset statistics file `%s` does not exist; fallback to current dataset statistics.",
            stats_path,
        )
        return None, None

    try:
        with open(stats_path, "r", encoding="utf-8") as f_json:
            dataset_statistics = json.load(f_json)
    except Exception as exc:
        logger.warning(
            "Failed to load pretrained dataset statistics from `%s`: %s. Fallback to current dataset statistics.",
            stats_path,
            exc,
        )
        return None, None

    if not isinstance(dataset_statistics, dict):
        logger.warning(
            "Pretrained dataset statistics from `%s` is not a dict; fallback to current dataset statistics.",
            stats_path,
        )
        return None, None

    return dataset_statistics, stats_path


def _load_dataset_statistics_override_for_training(cfg) -> Optional[dict]:
    trainer_cfg = getattr(cfg, "trainer", None)
    use_pretrained_statistics = bool(_cfg_get(trainer_cfg, "use_pretrained_dataset_statistics", False))
    pretrained_checkpoint = _cfg_get(trainer_cfg, "pretrained_checkpoint", None)

    if not use_pretrained_statistics:
        logger.info("Using current training dataset statistics (pretrained statistics override disabled).")
        return None

    if not pretrained_checkpoint:
        logger.warning(
            "`trainer.use_pretrained_dataset_statistics=true` but no `trainer.pretrained_checkpoint` was provided; "
            "fallback to current dataset statistics."
        )
        return None

    pretrained_statistics, pretrained_stats_path = _load_pretrained_dataset_statistics(pretrained_checkpoint)
    if pretrained_statistics is None:
        return None

    logger.info("Using pretrained dataset statistics from `%s` for training transforms.", pretrained_stats_path)
    return pretrained_statistics


def _build_dino_pixel_decoder_collator(cfg, *, training: bool):
    from starVLA.dataloader.dino_pixel_decoder_collator import DinoPixelDecoderCollator

    image_resolution = int(_cfg_get(cfg.datasets.vla_data, "image_resolution", 256))
    collator = DinoPixelDecoderCollator(image_resolution=image_resolution)
    return collator.train() if training else collator.eval()


def build_dataloaders(cfg) -> tuple[DataLoader, Optional[DataLoader]]:
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder

    vla_dataset_cfg = cfg.datasets.vla_data
    train_split_all = bool(_cfg_get(vla_dataset_cfg, "train_split_all", True))
    train_mode = "all" if train_split_all else "train"
    batch_size = cfg.datasets.vla_data.per_device_batch_size
    balance_dataset_weights = bool(vla_dataset_cfg.get("balance_dataset_weights", True))
    framework_name = getattr(cfg.framework, "name", None)
    is_latent_world = str(framework_name).lower() == "lawam"
    is_dino_pixel_decoder = str(framework_name).lower() in {"dinopixeldecoder", "dino_pixel_decoder"}
    train_num_workers = int(vla_dataset_cfg.get("num_workers", 8))
    val_num_workers = int(vla_dataset_cfg.get("val_num_workers", 4))
    prefetch_factor = int(vla_dataset_cfg.get("prefetch_factor", 2))
    pin_memory = True
    persistent_workers = bool(vla_dataset_cfg.get("persistent_workers", train_num_workers > 0))
    in_order = bool(vla_dataset_cfg.get("in_order", False))
    drop_last = bool(vla_dataset_cfg.get("drop_last", True))
    dataset_statistics_override = _load_dataset_statistics_override_for_training(cfg)

    collate = collate_fn
    if is_latent_world:
        policy_cfg = LatentWorldPolicyConfigBuilder(cfg).build()
        collate = _build_latent_world_collator(cfg, policy_cfg=policy_cfg, training=True)
    elif is_dino_pixel_decoder:
        collate = _build_dino_pixel_decoder_collator(cfg, training=True)

    vla_train_dataset = get_vla_dataset(
        data_cfg=vla_dataset_cfg,
        mode=train_mode,
        balance_dataset_weights=balance_dataset_weights,
        framework_name=framework_name,
        dataset_statistics_override=dataset_statistics_override,
    )
    train_dataset_statistics = dataset_statistics_override or vla_train_dataset.build_dataset_statistics()

    train_loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate,
        "num_workers": train_num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if train_num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        train_loader_kwargs["persistent_workers"] = persistent_workers
        train_loader_kwargs["worker_init_fn"] = _vla_worker_init_fn
        train_loader_kwargs["in_order"] = in_order
    vla_train_dataloader = DataLoader(vla_train_dataset, **train_loader_kwargs)

    vla_val_dataloader = None
    try:
        val_collate = collate_fn
        if is_latent_world:
            val_collate = _build_latent_world_collator(cfg, policy_cfg=policy_cfg, training=False)
        elif is_dino_pixel_decoder:
            val_collate = _build_dino_pixel_decoder_collator(cfg, training=False)
        vla_val_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            mode="val",
            balance_dataset_weights=balance_dataset_weights,
            framework_name=framework_name,
            dataset_statistics_override=train_dataset_statistics,
        )
        val_loader_kwargs = {
            "batch_size": batch_size,
            "collate_fn": val_collate,
            "num_workers": val_num_workers,
            "pin_memory": pin_memory,
            "drop_last": False,
        }
        if val_num_workers > 0:
            val_loader_kwargs["prefetch_factor"] = prefetch_factor
            val_loader_kwargs["persistent_workers"] = persistent_workers
            val_loader_kwargs["worker_init_fn"] = _vla_worker_init_fn
            val_loader_kwargs["in_order"] = in_order
        vla_val_dataloader = DataLoader(vla_val_dataset, **val_loader_kwargs)
    except ValueError as exc:
        logger.warning(f"Validation dataset unavailable, continue without val loader: {exc}")

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        output_path = Path(cfg.output_dir) / "dataset_statistics.json"
        save_dataset_statistics(train_dataset_statistics, output_path)
    return vla_train_dataloader, vla_val_dataloader


def build_dataloader(cfg):
    return build_dataloaders(cfg)[0]
