"""Build matched baseline/orange configs from the endpoint checkpoint's exact config.

Orange normalization uses the dataset's ordered training episodes only, never val.
The generated configs and pose buffers travel with new checkpoints.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pyarrow.parquet as pq

from scripts.prepare_photograph_object_pose import prepare
from starVLA.object_pose import ObjectPoseStore

BASE = Path('/data/home/zyh/LaWAM/results/Checkpoints/dexjoco_bimanual_photograph_C_h30_tip_endpoint/20261015_142505+dexjoco_bimanual_photograph_C_h30_tip_endpoint_sft')


def training_pose_statistics(dataset_root, sidecar_root, val_tail_ratio, allowed_episodes=None):
    # Same sorted metadata files/row order as LeRobotDataset._load_episodes_hf.
    files = sorted((dataset_root / 'meta/episodes').rglob('*.parquet'))
    episodes = [int(x) for p in files for x in pq.read_table(p, columns=['episode_index'])['episode_index'].to_pylist()]
    if allowed_episodes is not None:
        episodes = [ep for ep in episodes if ep in set(allowed_episodes)]
    if not episodes or len(set(episodes)) != len(episodes) or not 0 <= val_tail_ratio < 1:
        raise ValueError('Invalid episode metadata or validation ratio')
    n_val = min(max(1, int(np.floor(len(episodes) * val_tail_ratio))), len(episodes)-1) if len(episodes) >= 2 else 0
    training = episodes[:len(episodes)-n_val]
    data_files = sorted((dataset_root / 'data').rglob('*.parquet'))
    direct = 'object_pose_condition' in pq.read_schema(data_files[0]).names
    if direct:
        from starVLA.object_pose import validate_object_pose
        columns = ['episode_index', 'object_pose_condition', 'object_pose_valid']
        import pyarrow as pa
        table = pa.concat_tables([pq.read_table(p, columns=columns) for p in data_files])
        ids = table['episode_index'].to_numpy()
        values = validate_object_pose(np.stack(table['object_pose_condition'].to_pylist()))
        valid = table['object_pose_valid'].to_numpy()
    else:
        store = ObjectPoseStore(sidecar_root)
        meta = store.manifest(dataset_root.name)
    positions = []
    for episode in training:
        # frame() validates complete episode shape/indices/direction on first access.
        if direct:
            mask = ids == episode
            if not mask.any() or not valid[mask].all():
                raise ValueError(f'Invalid or missing pose in selected training episode {episode}')
            positions.append(values[mask, :3].astype(np.float64))
        else:
            store.frame(dataset_root.name, episode, 0, meta['fps'])
            positions.append(store.cache[(dataset_root.name, episode)][:, :3].astype(np.float64))
    positions = np.concatenate(positions)
    return dict(mean=positions.mean(0).tolist(), std=np.maximum(positions.std(0), 1e-6).tolist(),
                training_episodes=training, validation_episodes=episodes[len(training):], count=len(positions))


def configure(mode, output, sidecar_root, dataset_root, steps=10000, seed=2026):
    cfg = OmegaConf.load(BASE / 'config.yaml')
    for key in ('output_dir', 'log_dir'):
        cfg.pop(key, None)
    cfg.run_id = f'C_h30_pose_action_{mode}_{steps}_seed{seed}'
    cfg.run_root_dir = f'results/Checkpoints/pose_action_{mode}'
    cfg.seed = seed
    cfg.trackers = []
    cfg.is_debug = False
    am = cfg.framework.action_model
    am.action_expert_only = True
    am.enable_loss_distill = False
    am.perceptual_weight = 0.0
    am.lam_encoder_distill_weight = 0.0
    # Retain fingertip_loss_weight solely to instantiate/load the endpoint head;
    # action_expert_only skips its targets/loss and freezes it in eval mode.
    am.object_pose_sidecar_root = str(sidecar_root)
    am.flow_cfg.pose_conditioning = mode == 'orange'
    am.flow_cfg.pose_every_n_blocks = 1
    am.flow_cfg.pose_attention_heads = am.flow_cfg.attention_heads
    am.flow_cfg.pose_encoder_hidden_dim = 128
    cfg.datasets.vla_data.data_root_dir = str(dataset_root.parent)
    cfg.datasets.vla_data.pop('fingertip_sidecar_root', None)
    cfg.datasets.vla_data.train_split_all = False
    cfg.trainer.pretrained_checkpoint = str(BASE / 'final_model/pytorch_model.pt')
    cfg.trainer.load_pretrained_policy_flow = True
    cfg.trainer.load_pretrained_policy_lam = True
    cfg.trainer.use_pretrained_dataset_statistics = True
    cfg.trainer.max_train_steps = steps
    cfg.trainer.num_warmup_steps = min(100, max(1, steps // 10))
    cfg.trainer.save_interval = 1000
    cfg.trainer.eval_interval = 1000
    cfg.trainer.eval_batches = 5
    # Same lr for all action expert parameters in both runs.
    cfg.trainer.learning_rate = {'base': 1e-5}
    cfg.trainer.freeze = dict(freeze_vision_backbone=True, freeze_llm_backbone=True,
                              freeze_embedding=True, freeze_last_llm_layer=True,
                              keep_llm_first_n_layers=16, unfreeze_vision_merger=False,
                              unfreeze_lam_decoder=False, unfreeze_llm_last_n_layers=-1)
    cfg.trainer.ddp_find_unused_parameters = False
    episode_file = dataset_root / 'pose_train_episode_indices.json'
    allowed_episodes = json.loads(episode_file.read_text()) if episode_file.exists() else None
    cfg.datasets.vla_data.load_object_pose = mode == 'orange' and episode_file.exists()
    if allowed_episodes is not None:
        cfg.datasets.vla_data.object_pose_episode_indices = allowed_episodes
    if mode == 'orange' or allowed_episodes is not None:
        if not cfg.datasets.vla_data.load_object_pose and allowed_episodes is None:
            prepare(dataset_root, sidecar_root, validate_only=True)
        stats = training_pose_statistics(dataset_root, sidecar_root, cfg.datasets.vla_data.val_tail_ratio, allowed_episodes)
        if mode == 'orange':
            am.flow_cfg.pose_position_mean = stats['mean']
            am.flow_cfg.pose_position_std = stats['std']
        cfg.pose_data_provenance = stats
        if episode_file.exists():
            cfg.pose_data_provenance.dataset_provenance = str(dataset_root / 'object_pose_provenance.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Refusing to replace generated config: {output}')
    OmegaConf.save(cfg, output)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['baseline','orange'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sidecar-root', type=Path, default=Path('results/object_pose_sidecars'))
    parser.add_argument('--dataset-root', type=Path, default=Path('/data/home/zyh/dexjoco/outputs/photograph_object_pose_20261022/lerobot/bimanual_photograph'))
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=2026)
    configure(**vars(parser.parse_args()))
