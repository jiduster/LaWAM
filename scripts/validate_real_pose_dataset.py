"""Validate audited real data through the loader/collator and a tiny CPU backend.

This checks integration; it does not train or evaluate the 7 GB base policy.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.test_external_latent_action_head import build_tiny_backend, collate
from starVLA.dataloader.lerobot_datasets import get_vla_dataset

CONFIG_ROOT = Path('starVLA/config/training')


def load(mode, split):
    cfg=OmegaConf.load(CONFIG_ROOT/f'starvla_train_dexjoco_photograph_pose_action_{mode}.yaml')
    source=Path(cfg.trainer.pretrained_checkpoint).parents[1]
    stats=json.loads((source/'dataset_statistics.json').read_text())
    return cfg,get_vla_dataset(cfg.datasets.vla_data,mode=split,seed=cfg.seed,
                              framework_name='LaWAM',dataset_statistics_override=stats)


def main():
    torch.set_num_threads(1)
    cfg,orange=load('orange','train')
    _,baseline=load('baseline','train')
    _,validation=load('orange','val')
    train_ids=orange.datasets[0]._all_trajectory_ids.tolist()
    active_ids=orange.datasets[0]._active_episode_indices.tolist()
    val_ids=validation.datasets[0]._active_episode_indices.tolist()
    assert len(train_ids)==87 and len(active_ids)==86 and val_ids==[99]
    assert set(active_ids).isdisjoint(val_ids)
    assert active_ids==baseline.datasets[0]._active_episode_indices.tolist()
    assert active_ids==list(cfg.pose_data_provenance.training_episodes)
    # First and last frame from every allowed episode: read the real data column,
    # preserve original IDs, and reject quarantined rows explicitly.
    single=orange.datasets[0]
    for episode in train_ids:
        meta=single.episodes_hf[single._episode_index_to_row[episode]]
        for index in (0,int(meta['length'])-1):
            assert single.get_object_pose_condition(episode,index).shape==(12,)
    try:
        single.get_object_pose_condition(15,0)
    except ValueError:
        pass
    else:
        raise AssertionError('Quarantined pose was accepted')
    samples=[]
    for index in range(2):
        a,b=orange[index],baseline[index]
        for key in ('episode_index','frame_index','dataset_name'):
            assert a[key]==b[key]
        for key in ('primary_videos','wrist_images','state','action'):
            torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
        assert a['object_pose_condition'].shape==(12,)
        assert 'object_pose_condition' not in b
        samples.append(a)
    backend=build_tiny_backend(mode='vlm_queries',pose=True,action_only=True,fingertip_mode='endpoint')
    backend.model_cfg.action_horizon=30
    backend.model_cfg.future_action_window_size=29
    backend.model_cfg.flow_cfg.horizon_sec=1.
    backend.flow.action_horizon=30
    backend.model_cfg.repeated_diffusion_steps=2
    # Use the actual training-only mean/std in the model as well.
    backend.flow.pose_encoder.position_mean.copy_(torch.tensor(list(cfg.framework.action_model.flow_cfg.pose_position_mean)))
    backend.flow.pose_encoder.position_std.copy_(torch.tensor(list(cfg.framework.action_model.flow_cfg.pose_position_std)))
    batch=collate(backend,samples)
    np.testing.assert_array_equal(batch['object_pose_condition'].numpy(),np.stack([s['object_pose_condition'].numpy() for s in samples]))
    optimizer=torch.optim.AdamW([p for p in backend.parameters() if p.requires_grad],lr=.003)
    losses=[]
    for step in range(3):
        torch.manual_seed(2026)
        optimizer.zero_grad(set_to_none=True)
        loss=backend(batch=batch)['loss_flow']
        loss.backward()
        if step==1:
            assert backend.flow.pose_encoder.groups[0][0].weight.grad.abs().sum()>0
        optimizer.step(); losses.append(float(loss))
    assert np.isfinite(losses).all()
    print('REAL_POSE_DATA_OK '+json.dumps(dict(allowed_episodes=len(train_ids),train_episodes=len(active_ids),
          validation_episodes=val_ids,train_frames=cfg.pose_data_provenance.count,
          sampled=[(s['episode_index'],s['frame_index']) for s in samples],tiny_cpu_losses=losses)))


if __name__=='__main__':
    main()
