"""Validate Sharpa H30 configs with real train/val data and the production collator.

Run from the repository root with PYTHONPATH=. and the lawam Python environment.
Only the local Qwen processor is loaded; no policy checkpoint or GPU is needed.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader import _build_latent_world_collator
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
from starVLA.model.framework.vlas.flowmatching_expert import (
    ConditionalFlowMatchingHead,
    build_time_grid,
)


def validate(config_path):
    cfg = OmegaConf.load(config_path)
    policy = LatentWorldPolicyConfigBuilder(cfg).build()
    data_cfg = cfg.datasets.vla_data
    assert policy.action_horizon == data_cfg.action_horizon == 30
    assert policy.future_action_window_size == 29 and policy.past_action_window_size == 0
    assert policy.flow_cfg.horizon_sec == data_cfg.sec_chunk == 1.5
    assert policy.flow_cfg.action_dim == policy.flow_cfg.state_dim in (29, 58)
    assert not cfg.trainer.load_pretrained_policy_lam
    assert not cfg.trainer.use_pretrained_dataset_statistics
    collator = _build_latent_world_collator(cfg, policy_cfg=policy, training=False)
    report = {"config": str(config_path), "splits": []}
    for split in ("train", "val"):
        mixture = get_vla_dataset(data_cfg, mode=split, seed=cfg.seed, framework_name="LaWAM")
        assert len(mixture.datasets) == 1
        dataset = mixture.datasets[0]
        assert dataset.action_hz == 20
        ids = list(map(int, dataset.trajectory_ids))
        assert ids == (list(range(99)) if split == "train" else [99])
        for key, offsets in dataset.delta_indices.items():
            if key.startswith("action."):
                np.testing.assert_array_equal(offsets, np.arange(30))
            elif key.startswith("video."):
                np.testing.assert_array_equal(offsets, [0, 29])

        # Check both a complete H30 window and clamping at the episode end.
        episode = ids[0]
        length = int(dataset.trajectory_lengths[0])
        samples = []
        for frame in (0, length - 2):
            raw = dataset.get_step_data(episode, frame)
            transformed = dataset.transforms(raw)
            sample = mixture._build_output_sample(
                dataset, transformed, dataset.modality_keys["video"], episode, frame
            )
            assert sample["action"].shape == (30, policy.flow_cfg.action_dim)
            assert sample["state"].shape == (1, policy.flow_cfg.state_dim)
            assert sample["primary_videos"].shape == (1, 2, 3, 256, 256)
            expected_wrists = 1 if policy.flow_cfg.action_dim == 29 else 2
            assert sample["wrist_images"].shape == (expected_wrists, 3, 256, 256)
            assert sample["lang"] and "__obj_" not in sample["lang"]
            if frame == length - 2:
                # All offsets after the final real step must repeat its absolute target.
                torch.testing.assert_close(
                    sample["action"][1:], sample["action"][1:2].expand(29, -1)
                )
                last_raw = dataset.get_step_data(episode, length - 1)
                last = dataset.transforms(last_raw)
                torch.testing.assert_close(sample["action"][-1], last["action"][0])
                for key in dataset.modality_keys["video"]:
                    # Wrist streams contain only the current image, not a future target.
                    if "wrist" not in key:
                        torch.testing.assert_close(transformed[key][-1], last[key][0])
            samples.append(sample)

        collator.train() if split == "train" else collator.eval()
        batch = collator(samples)
        assert batch["actions"].shape == (2, 30, policy.flow_cfg.action_dim)
        assert batch["primary_video"].shape == (2, 2, 3, 256, 256)
        assert batch["actions_mask"].all() and batch["state_mask"].all()
        for i, sample in enumerate(samples):
            torch.testing.assert_close(batch["actions"][i], sample["action"])
            torch.testing.assert_close(batch["state"][i], sample["state"][0])
        grid, valid, counts = build_time_grid(policy.flow_cfg.horizon_sec, batch["action_hz"], 30)
        assert counts.tolist() == [30, 30] and valid.all()
        torch.testing.assert_close(grid[:, -1], torch.full((2,), 1.45))

        # Exercise the actual flow loss/time-mask contract, with a small CPU model.
        # VLM/LAM features here are synthetic; the actions/states are the real batch.
        flow_cfg = copy.deepcopy(policy.flow_cfg)
        flow_cfg.vlm_dim = flow_cfg.hidden_dim = 32
        flow_cfg.vision_dim, flow_cfg.num_vision_tokens = 8, 4
        flow_cfg.num_layers, flow_cfg.attention_heads = 2, 4
        flow = ConditionalFlowMatchingHead(flow_cfg)
        flow.action_horizon = 30
        loss = flow(
            h_t=torch.randn(2, 4, 8), h_t1_star=torch.randn(2, 4, 8),
            h_vlm=torch.randn(2, 6, 32),
            **{key: batch[key] for key in (
                "state", "actions", "action_hz", "embodiment_id", "state_mask", "actions_mask"
            )},
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in flow.parameters())
        report["splits"].append({
            "split": split, "episodes": len(ids), "actions": list(batch["actions"].shape),
            "state": list(batch["state"].shape), "action_offsets": [0, 29],
            "video_offsets": [0, 29], "fps": dataset.action_hz,
            "visual_interval_sec": 1.45, "flow_loss": float(loss.detach()),
            "language": samples[0]["lang"], "episode_end_clamping": "passed",
        })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    result = validate(args.config)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("SHARPA_H30_OK " + json.dumps(result, ensure_ascii=False), flush=True)
