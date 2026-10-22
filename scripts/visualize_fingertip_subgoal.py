#!/usr/bin/env python3
"""Open-loop fingertip overlay for one DexJoCo photograph trajectory.

Each chunk is the training pair (frame t, frame t+29). The endpoint head reads
mean-pooled future DINO tokens and returns 8 fingertip positions in the fixed
ego camera frame. Those points, and the sidecar GT at t+29, are drawn on the
future ego image.

Two readouts are shown:
  predicted DINO  -> the subgoal the world model actually emitted
  true future DINO -> whether the same MLP can read fingertips when the feature is correct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.dataloader import _build_latent_world_collator
from starVLA.model.framework.base_framework import baseframework


TIP_NAMES = ("Lff", "Lmf", "Lrf", "Lth", "Rff", "Rmf", "Rrf", "Rth")
TIP_COLORS = (
    (255, 70, 70),
    (255, 150, 30),
    (230, 210, 40),
    (80, 220, 90),
    (70, 170, 255),
    (60, 90, 255),
    (180, 80, 255),
    (255, 80, 190),
)
# Fixed DexJoCo ego camera is XML camera "back": 640x640, vertical FOV 45 deg.
# Sidecar coordinates are optical, with +Y toward the bottom of the image.
# Checked by projecting episode-0 GT tips onto the ego frame.
IMAGE_SIZE = 640
FOVY_DEG = 45.0


def _project_tips(points_m: np.ndarray) -> np.ndarray:
    if points_m.shape != (8, 3):
        raise ValueError(f"Expected tips [8,3], got {points_m.shape}")
    fovy = np.deg2rad(FOVY_DEG)
    focal = (IMAGE_SIZE / 2.0) / np.tan(fovy / 2.0)
    center = (IMAGE_SIZE - 1) / 2.0
    z = np.clip(points_m[:, 2], 1e-4, None)
    u = center + focal * points_m[:, 0] / z
    v = center + focal * points_m[:, 1] / z
    return np.stack([u, v], axis=1)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_pair(
    image_rgb: np.ndarray,
    gt_m: np.ndarray,
    pred_m: np.ndarray,
    *,
    title: str,
) -> Image.Image:
    canvas = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    gt_uv = _project_tips(gt_m)
    pred_uv = _project_tips(pred_m)
    for color, gt, pred in zip(TIP_COLORS, gt_uv, pred_uv):
        draw.line((float(gt[0]), float(gt[1]), float(pred[0]), float(pred[1])), fill=color, width=2)
        r = 7
        draw.ellipse((pred[0] - r, pred[1] - r, pred[0] + r, pred[1] + r), outline=color, width=3)
        draw.ellipse((gt[0] - 5, gt[1] - 5, gt[0] + 5, gt[1] + 5), fill=color, outline=(0, 0, 0))
    font = _font(16)
    draw.rectangle((0, 0, IMAGE_SIZE, 28), fill=(0, 0, 0))
    draw.text((8, 5), title, fill=(255, 255, 255), font=font)
    return canvas


def _caption_bar(width: int, lines: list[str]) -> Image.Image:
    font = _font(18)
    height = 8 + 24 * len(lines)
    bar = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(bar)
    for i, line in enumerate(lines):
        draw.text((10, 6 + 24 * i), line, fill=(240, 240, 240), font=font)
    return bar


def _stack_horizontal(images: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def _endpoint_meters(backend, features: torch.Tensor) -> np.ndarray:
    summary = features.float().mean(dim=1)
    pred = backend.fingertip_head(summary).view(summary.shape[0], 8, 3)
    center = backend._fingertip_endpoint_center(summary)
    scale = float(backend.model_cfg.fingertip_target_scale)
    return (pred * scale + center).detach().float().cpu().numpy()


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _future_rgb(raw_primary: np.ndarray) -> np.ndarray:
    frame = np.asarray(raw_primary[-1])
    if frame.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Expected future ego frame {(IMAGE_SIZE, IMAGE_SIZE, 3)}, got {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _tip_metrics(pred_m: np.ndarray, gt_m: np.ndarray) -> dict[str, float]:
    delta = pred_m - gt_m
    per_tip_mm = np.linalg.norm(delta, axis=-1) * 1000.0
    rmse_mm = float(np.sqrt(np.mean(delta ** 2)) * 1000.0)
    return {
        "rmse_mm": rmse_mm,
        "mean_l2_mm": float(per_tip_mm.mean()),
        "max_l2_mm": float(per_tip_mm.max()),
        "per_tip_l2_mm": [float(value) for value in per_tip_mm],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "results/Checkpoints/dexjoco_bimanual_photograph_C_h30_tip_endpoint/"
            "20261015_142505+dexjoco_bimanual_photograph_C_h30_tip_endpoint_sft/final_model/pytorch_model.pt"
        ),
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/vis/fingertip_subgoal_ep000"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    run_dir = checkpoint.parents[1]
    cfg = OmegaConf.load(run_dir / "config.yaml")
    stats_path = run_dir / "dataset_statistics.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None

    print(f"loading dataset from {cfg.datasets.vla_data.data_root_dir}", flush=True)
    dataset = make_LeRobotSingleDataset(
        data_root_dir=cfg.datasets.vla_data.data_root_dir,
        data_name="bimanual_photograph",
        robot_type="dexjoco_bimanual",
        mode="all",
        data_cfg=cfg.datasets.vla_data,
        framework_name="LaWAM",
        dataset_statistics_override=stats,
    )
    dataset.transforms.eval()
    traj_index = dataset.get_trajectory_index(int(args.episode))
    episode_length = int(dataset.trajectory_lengths[traj_index])
    horizon = int(cfg.framework.action_model.action_horizon)
    starts = list(range(0, episode_length - horizon, horizon))
    if not starts:
        raise RuntimeError(f"Episode {args.episode} length {episode_length} cannot fit horizon {horizon}.")
    print(f"episode {args.episode} length {episode_length} chunks {starts}", flush=True)

    print(f"loading model {checkpoint}", flush=True)
    model = baseframework.from_pretrained(str(checkpoint))
    model.eval()
    device = torch.device(args.device)
    model.to(device)
    backend = model.policy_backend
    if backend.fingertip_head is None or str(backend.model_cfg.fingertip_target_mode).lower() != "endpoint":
        raise RuntimeError("Checkpoint has no endpoint fingertip head.")
    collator = _build_latent_world_collator(cfg, policy_cfg=backend.model_cfg, training=False)
    video_keys = list(dataset.modality_keys["video"])

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    sheets = []
    for start in starts:
        future = start + horizon - 1
        raw = dataset.get_step_data(int(args.episode), start, modality_keys_override={"video": video_keys})
        future_rgb = _future_rgb(np.array(raw["video.primary_view"], copy=True))
        transformed = dataset.transforms(raw)
        sample = LeRobotMixtureDataset._build_latent_world_output_sample(
            LeRobotMixtureDataset.__new__(LeRobotMixtureDataset),
            dataset,
            transformed,
            video_keys,
            int(args.episode),
            start,
        )
        batch = _move_batch(collator([sample]), device)
        with torch.inference_mode():
            shared = backend._run_shared_encoding_train(
                prepared_batch=batch,
                source="visualize_fingertip_subgoal",
                lam_features_with_no_grad=True,
            )
            pred_m = _endpoint_meters(backend, shared.h_t1_pred)[0]
            probe_m = _endpoint_meters(backend, shared.h_t1_gt)[0]
        gt_positions = sample["fingertip_positions"][-1]
        gt_m = gt_positions.detach().cpu().numpy() if torch.is_tensor(gt_positions) else np.asarray(gt_positions)
        pred_metrics = _tip_metrics(pred_m, gt_m)
        probe_metrics = _tip_metrics(probe_m, gt_m)
        subgoal = _draw_pair(
            future_rgb,
            gt_m,
            pred_m,
            title=f"subgoal  RMSE {pred_metrics['rmse_mm']:.1f} mm",
        )
        probe = _draw_pair(
            future_rgb,
            gt_m,
            probe_m,
            title=f"true-DINO probe  RMSE {probe_metrics['rmse_mm']:.1f} mm",
        )
        pair = _stack_horizontal([subgoal, probe])
        caption = _caption_bar(
            pair.width,
            [
                f"episode {args.episode}   current t={start}   subgoal frame t={future}   filled=GT  ring=readout",
                (
                    f"subgoal mean L2 {pred_metrics['mean_l2_mm']:.1f} mm"
                    f"    probe mean L2 {probe_metrics['mean_l2_mm']:.1f} mm"
                ),
            ],
        )
        sheet = Image.new("RGB", (pair.width, pair.height + caption.height))
        sheet.paste(caption, (0, 0))
        sheet.paste(pair, (0, caption.height))
        image_path = output_dir / f"chunk_t{start:04d}.png"
        sheet.save(image_path)
        sheets.append(sheet.resize((sheet.width // 2, sheet.height // 2), Image.Resampling.BILINEAR))
        row = {
            "start": start,
            "future": future,
            "image": str(image_path),
            "subgoal": pred_metrics,
            "true_dino_probe": probe_metrics,
        }
        rows.append(row)
        print(
            f"t={start:4d}->{future:4d}  subgoal RMSE {pred_metrics['rmse_mm']:.1f} mm"
            f"  probe RMSE {probe_metrics['rmse_mm']:.1f} mm",
            flush=True,
        )

    overview = Image.new("RGB", (sheets[0].width, sum(image.height for image in sheets)))
    y = 0
    for image in sheets:
        overview.paste(image, (0, y))
        y += image.height
    overview_path = output_dir / "overview.png"
    overview.save(overview_path)
    summary = {
        "checkpoint": str(checkpoint),
        "episode": int(args.episode),
        "episode_length": episode_length,
        "horizon": horizon,
        "coordinate_frame": "dexjoco_fixed_ego_camera_optical",
        "projection": {"width": IMAGE_SIZE, "height": IMAGE_SIZE, "fovy_deg": FOVY_DEG, "y_axis": "image_down"},
        "chunks": rows,
        "subgoal_rmse_mm_mean": float(np.mean([row["subgoal"]["rmse_mm"] for row in rows])),
        "probe_rmse_mm_mean": float(np.mean([row["true_dino_probe"]["rmse_mm"] for row in rows])),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {overview_path}", flush=True)
    print(
        f"mean subgoal RMSE {summary['subgoal_rmse_mm_mean']:.1f} mm"
        f"    mean probe RMSE {summary['probe_rmse_mm_mean']:.1f} mm",
        flush=True,
    )


if __name__ == "__main__":
    main()
