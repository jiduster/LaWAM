"""Resolve portable green Stage 2 assets and validate data before launching DDP."""
import argparse
import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf
import pyarrow.parquet as pq

from scripts.configure_pose_action import training_pose_statistics
from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder


REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / 'starVLA/config/training/starvla_train_dexjoco_photograph_pose_plan_stage2.yaml'


def configure(*, data_root, fingertip_root, asset_root, output_root, output, steps=40000, seed=2026):
    data_root = Path(data_root).expanduser().resolve()
    fingertip_root = Path(fingertip_root).expanduser().resolve()
    asset_root = Path(asset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if steps < 1:
        raise ValueError('steps must be positive')
    if data_root.name != 'bimanual_photograph':
        raise ValueError('--data-root must be the bimanual_photograph dataset directory')
    if output.exists() or output.with_suffix('.lam.yaml').exists():
        raise FileExistsError(f'Refusing to replace generated config: {output}')
    cfg = OmegaConf.load(TEMPLATE)
    am = cfg.framework.action_model
    required_files = [data_root / 'meta/info.json', data_root / 'pose_train_episode_indices.json',
                      data_root / 'object_pose_provenance.json']
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    episodes = json.loads((data_root / 'pose_train_episode_indices.json').read_text())
    if len(episodes) != 87 or len(set(episodes)) != 87:
        raise ValueError('Expected the original audited set of 87 unique episodes')
    if float(json.loads((data_root / 'meta/info.json').read_text())['fps']) != 30:
        raise ValueError('Green H30 recipe requires dataset fps=30')
    # Copy videos with symlinks dereferenced when migrating. Catch dangling links now.
    videos = list((data_root / 'videos').rglob('*'))
    if not any(p.is_file() for p in videos):
        raise FileNotFoundError(f'No readable videos in {data_root / "videos"}')
    broken = [str(p) for p in videos if p.is_symlink() and not p.exists()]
    if broken:
        raise FileNotFoundError(f'Broken dataset video links: {broken[:3]}')
    sidecar = fingertip_root / data_root.name
    manifest = json.loads((sidecar / 'manifest.json').read_text())
    entries = {int(e['episode_index']): e for e in manifest['episodes']}
    for ep in episodes:
        if ep not in entries or not (sidecar / entries[ep]['file']).is_file():
            raise FileNotFoundError(f'Missing endpoint sidecar for episode {ep} in {sidecar}')
    parquet_files = sorted((data_root / 'data').rglob('*.parquet'))
    if not parquet_files or 'object_pose_condition' not in pq.read_schema(parquet_files[0]).names:
        raise ValueError('Dataset must contain the audited object_pose_condition column')
    stats = training_pose_statistics(data_root, Path(''), cfg.datasets.vla_data.val_tail_ratio, episodes)
    cfg.datasets.vla_data.data_root_dir = str(data_root.parent)
    cfg.datasets.vla_data.fingertip_sidecar_root = str(fingertip_root)
    cfg.datasets.vla_data.object_pose_episode_indices = episodes
    am.plan_pose_position_mean = stats['mean']
    am.plan_pose_position_std = stats['std']
    cfg.pose_data_provenance = stats
    cfg.pose_data_provenance.dataset_provenance = str(data_root / 'object_pose_provenance.json')
    cfg.framework.qwenvl.base_vlm = str(asset_root / cfg.framework.qwenvl.base_vlm)
    am.lam_ckpt_path = str(asset_root / am.lam_ckpt_path)
    cfg.trainer.pretrained_checkpoint = str(asset_root / cfg.trainer.pretrained_checkpoint)
    lam_source = asset_root / am.lam_yaml_path
    for path in (Path(cfg.framework.qwenvl.base_vlm), Path(am.lam_ckpt_path),
                 Path(cfg.trainer.pretrained_checkpoint), lam_source):
        if not path.exists():
            raise FileNotFoundError(f'Missing model asset: {path}')
    # The released LAM yaml contains a machine-specific DINO location.
    lam = OmegaConf.load(lam_source)
    dino = asset_root / 'results/Checkpoints/dinov3-vitb16-pretrain-lvd1689m'
    if not (dino / 'config.json').is_file():
        raise FileNotFoundError(f'Missing DINO model directory: {dino}')
    lam.model.vision_model_id = str(dino)
    am.lam_yaml_path = str(output.with_suffix('.lam.yaml'))
    cfg.trainer.max_train_steps = steps
    cfg.trainer.num_warmup_steps = min(1000, max(1, steps // 10))
    cfg.seed = seed
    cfg.run_id = f'C_h30_tip_endpoint_pose_green_{steps}_seed{seed}'
    cfg.run_root_dir = str(output_root)
    # Model code validates known fields/windows before any GPU process is launched.
    policy = LatentWorldPolicyConfigBuilder(cfg).build()
    assert policy.plan_pose_conditioning and not policy.flow_cfg.pose_conditioning
    assert policy.action_horizon == 30 and not policy.action_expert_only
    output_root.mkdir(parents=True, exist_ok=True)
    # Each model is below 7.6 GB; include final and one-file slack.
    expected_bytes = (steps // int(cfg.trainer.save_interval) + 2) * 7_600_000_000
    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes < expected_bytes:
        raise OSError(f'Output needs about {expected_bytes / 1e9:.1f} GB; '
                      f'only {free_bytes / 1e9:.1f} GB free at {output_root}')
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(lam, output.with_suffix('.lam.yaml'))
    OmegaConf.save(cfg, output)
    print(f'GREEN_CONFIG={output}')
    print(f'Green only; H=30, action/state=44/46, train/val=86/1, steps={steps}, '
          f'per-GPU batch=4, accumulation=1. Launch 8 ranks for global batch=32.')
    print(f'Output={output_root}; model-file budget={expected_bytes / 1e9:.1f} GB')
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--fingertip-root', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, default=REPO)
    parser.add_argument('--output-root', type=Path, default=REPO / 'results/pose_stage2_green_local')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--seed', type=int, default=2026)
    configure(**vars(parser.parse_args()))
