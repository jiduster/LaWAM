"""Preview unsupervised DINO anchor proposals without loading the VLM/action expert."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / (
        "starVLA/config/training/starvla_train_dexjoco_bimanual_photograph_C_h30_visual_anchors.yaml"
    ))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, nargs="+")
    source.add_argument("--video", type=Path)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 120, 240])
    parser.add_argument("--checkpoint", type=Path, help="Override config initializer for the frozen DINO weights.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if any(index < 0 for index in args.frames):
        parser.error("--frames must be nonnegative.")

    from latent_action_model.core.vjepa_encoder import build_vision_encoder
    from starVLA.model.framework.latent_world.batch_utils import imagenet_normalize_image_batch_, prepare_frame_spatial_uint8
    from starVLA.model.framework.vlas.semantic_visual_anchors import SemanticVisualAnchors

    cfg = OmegaConf.load(args.config)
    policy = cfg.framework.action_model
    lam_cfg = OmegaConf.load(policy.lam_yaml_path).model
    layers = lam_cfg.get("latent_layer_to_use", -2)
    num_layers = 1 if isinstance(layers, int) else len(layers)
    encoder, dimension = build_vision_encoder(
        str(lam_cfg.vision_model_id), num_latent_layers=num_layers,
        norm_layer_type=str(lam_cfg.get("norm_latents_type", "ln")),
        enable_norm=bool(lam_cfg.get("norm_latents", True)),
    )
    checkpoint = args.checkpoint or Path(cfg.trainer.pretrained_checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "pytorch_model.pt"
    # Only touch frozen vision tensors in the large checkpoint; no full VLM allocation.
    weights = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    prefix = "policy_backend.lam.vision_encoder."
    vision_weights = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
    encoder.load_state_dict(vision_weights, strict=True)
    del weights, vision_weights
    encoder = encoder.to(args.device).eval()
    height, width = map(int, lam_cfg.image_hw)
    patch_size = int(lam_cfg.patch_size)
    head = SemanticVisualAnchors(
        vision_dim=dimension, vlm_dim=int(policy.flow_cfg.vlm_dim),
        state_dim=int(policy.flow_cfg.state_dim), token_dim=dimension,
        grid_height=height // patch_size, grid_width=width // patch_size,
        num_tokens=int(policy.visual_anchor_num_tokens), hidden_dim=int(policy.visual_anchor_hidden_dim),
        kmeans_iters=int(policy.visual_anchor_kmeans_iters), spatial_weight=float(policy.visual_anchor_spatial_weight),
    ).to(args.device).eval()

    def frames():
        if args.images:
            for index, path in enumerate(args.images):
                with Image.open(path) as image:
                    yield f"image_{index:04d}_{path.stem}", image.convert("RGB")
        else:
            import av

            remaining = set(args.frames)
            with av.open(str(args.video)) as container:
                for index, frame in enumerate(container.decode(video=0)):
                    if index in remaining:
                        yield f"frame_{index:06d}", frame.to_image().convert("RGB")
                        remaining.remove(index)
                    if not remaining:
                        break
            if remaining:
                raise ValueError(f"Video does not contain requested frame indices: {sorted(remaining)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    palette = ["#ef4444", "#22c55e", "#3b82f6", "#eab308", "#d946ef", "#06b6d4", "#f97316", "#84cc16"]
    report = []
    for name, image in frames():
        raw = prepare_frame_spatial_uint8(
            np.array(image, copy=True), target_hw=(height, width), apply_center_crop_90=False,
        )
        inputs = raw.float().unsqueeze(0).to(args.device) / 255
        imagenet_normalize_image_batch_(inputs)
        with torch.inference_mode():
            features = encoder.encode(inputs, n=-2)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            _, uv, indices = head.propose(features[:, 0])
        uv = uv[0].cpu()
        canvas = Image.fromarray(raw.permute(1, 2, 0).numpy()).resize((width * 3, height * 3))
        draw = ImageDraw.Draw(canvas)
        for index, point in enumerate(uv):
            x, y = ((point + 1) * torch.tensor(canvas.size) / 2).tolist()
            color = palette[index % len(palette)]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=1)
            draw.text((x + 7, y - 8), str(index), fill=color, stroke_width=1, stroke_fill="black")
        destination = args.output_dir / f"{name}.png"
        canvas.save(destination)
        report.append({"image": str(destination), "uv": uv.tolist(), "patch_indices": indices[0].cpu().tolist()})
        print(destination)
    (args.output_dir / "anchors.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
