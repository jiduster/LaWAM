"""Run real H30 samples through the green collator and a tiny CPU VLM/LAM backend."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.test_external_latent_action_head import build_tiny_backend
from starVLA.dataloader import _build_latent_world_collator
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def validate(config):
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    cfg = OmegaConf.load(config)
    assert cfg.framework.action_model.plan_pose_conditioning
    assert not cfg.framework.action_model.flow_cfg.pose_conditioning
    model = build_tiny_backend('vlm_queries', plan_pose=True, fingertip_mode='endpoint', layers=4)
    for key in ('action_horizon', 'future_action_window_size', 'past_action_window_size',
                'repeated_diffusion_steps', 'perceptual_weight', 'lam_encoder_distill_weight',
                'fingertip_loss_weight', 'detach_future_feature'):
        setattr(model.model_cfg, key, cfg.framework.action_model[key])
    model.model_cfg.flow_cfg.horizon_sec = cfg.framework.action_model.flow_cfg.horizon_sec
    model.flow.action_horizon = model.model_cfg.action_horizon
    model.plan_pose_encoder.position_mean.copy_(torch.tensor(list(cfg.framework.action_model.plan_pose_position_mean)))
    model.plan_pose_encoder.position_std.copy_(torch.tensor(list(cfg.framework.action_model.plan_pose_position_std)))
    collator = _build_latent_world_collator(cfg, policy_cfg=model.model_cfg, training=True)
    collator._processor = model.processor
    collator._placeholder_token_id = 99
    report = {'config': str(config), 'splits': []}
    for split in ('train', 'val'):
        dataset = get_vla_dataset(cfg.datasets.vla_data, mode=split, seed=cfg.seed, framework_name='LaWAM')
        single = dataset.datasets[0]
        ids = list(map(int, single._active_episode_indices))
        expected = cfg.pose_data_provenance.training_episodes if split == 'train' else cfg.pose_data_provenance.validation_episodes
        assert ids == list(expected)
        assert single.action_hz == 30
        for key, indices in single.delta_indices.items():
            if key.startswith('action.'):
                np.testing.assert_array_equal(indices, np.arange(30))
            elif key.startswith('video.'):
                np.testing.assert_array_equal(indices, [0, 29])
        samples = [dataset[i] for i in range(2)]
        batch = collator(samples)
        assert batch['actions'].shape == (2, 30, 44)
        assert batch['state'].shape == (2, 46)
        assert batch['object_pose_condition'].shape == (2, 12)
        assert batch['fingertip_targets'].shape == (2, 8, 3)
        for i,sample in enumerate(samples):
            torch.testing.assert_close(batch['object_pose_condition'][i], sample['object_pose_condition'])
            torch.testing.assert_close(batch['fingertip_targets'][i],
                                       torch.as_tensor(sample['fingertip_positions'][-1], dtype=torch.float32))
        if split == 'train':
            model.train()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            losses_seen = []
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                losses = model(batch=batch)
                expected_loss = (losses['loss_flow'] + .1*losses['loss_perceptual']
                                 + .1*losses['loss_distill'] + .1*losses['loss_fingertip'])
                torch.testing.assert_close(losses['loss_total'], expected_loss)
                assert torch.isfinite(expected_loss)
                losses['loss_total'].backward()
                for name, module in [('adapter', model.vlm_to_lam.pose_adapter), ('vlm', model.vlm),
                                     ('world_decoder', model.lam.decoder), ('endpoint', model.fingertip_head),
                                     ('flow', model.flow)]:
                    grad = sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)
                    assert grad > 0, (name, grad)
                if step == 1:
                    assert sum(p.grad.abs().sum().item() for p in model.plan_pose_encoder.parameters()
                               if p.grad is not None) > 0
                optimizer.step()
                losses_seen.append({k: float(v.detach()) for k,v in losses.items() if k.startswith('loss_')})
            report['tiny_training_losses'] = losses_seen
        report['splits'].append(dict(split=split, episodes=len(ids), actions=list(batch['actions'].shape),
                                      pose=list(batch['object_pose_condition'].shape),
                                      rows=[(int(s['episode_index']), int(s['frame_index'])) for s in samples]))
    print('REAL_GREEN_DATA_OK ' + json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Generated YAML from train_pose_plan.sh --dry-run')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = validate(args.config)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2))
