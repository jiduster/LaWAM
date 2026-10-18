"""Offline action diagnostics using privileged teacher latents and future images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FixedEpisodeWindows(Dataset):
    """Enumerate unique, full-horizon windows without mixture resampling/retries."""

    def __init__(self, mixture, *, dataset_name=None, episode_ids=None, windows_per_episode=64):
        self.mixture = mixture
        self.windows = []
        self.episodes = []
        self.skipped_episodes = []
        selected = set(episode_ids) if episode_ids is not None else None
        found = set()
        for dataset_index, dataset in enumerate(mixture.datasets):
            if dataset_name is not None and dataset.dataset_name != dataset_name:
                continue
            dataset.transforms.eval()
            offsets = np.concatenate([np.asarray(v) for v in dataset.delta_indices.values()])
            first = max(0, -int(offsets.min()))
            last_offset = int(offsets.max())
            video_keys = list(dataset.modality_keys['video'])
            if not video_keys:
                raise ValueError(f'No video keys for {dataset.dataset_name}.')
            video_offsets = dataset.delta_indices[video_keys[0]]
            if len(video_offsets) < 2 or video_offsets[-1] <= video_offsets[0]:
                raise ValueError('Current and future video frames must have different timestamps.')
            for episode_id, length in zip(dataset.trajectory_ids, dataset.trajectory_lengths):
                episode_id, length = int(episode_id), int(length)
                if selected is not None and episode_id not in selected:
                    continue
                found.add(episode_id)
                stop = length - last_offset
                count = stop - first
                info = {
                    'dataset': dataset.dataset_name, 'episode_id': episode_id, 'length': length,
                    'video_offsets': [int(v) for v in video_offsets],
                    'future_dt_sec': float(video_offsets[-1] - video_offsets[0]) / float(dataset.action_hz),
                }
                if count <= 0:
                    self.skipped_episodes.append(info)
                    continue
                n = count if windows_per_episode == 0 else min(count, windows_per_episode)
                starts = np.rint(np.linspace(first, stop - 1, n)).astype(int)
                info['windows'] = len(starts)
                self.episodes.append(info)
                for start in starts:
                    self.windows.append({
                        'dataset': dataset.dataset_name, 'dataset_index': dataset_index,
                        'episode_id': episode_id, 'start_frame': int(start),
                        'future_frame': int(start + video_offsets[-1]),
                    })
        if selected is not None and selected - found:
            raise ValueError(f'Episodes {sorted(selected - found)} are not in the requested split/dataset.')
        if not self.windows:
            raise ValueError('No complete windows in the requested split/dataset.')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        dataset = self.mixture.datasets[window['dataset_index']]
        video_keys = self.mixture._select_video_keys_for_sample(dataset, index)
        raw = dataset.get_step_data(
            window['episode_id'], window['start_frame'], modality_keys_override={'video': video_keys},
        )
        transform = self.mixture._get_transforms_for_selected_video_keys(dataset, video_keys)
        sample = self.mixture._build_output_sample(
            dataset, transform(raw), video_keys, window['episode_id'], window['start_frame'],
        )
        return sample


def resolve_checkpoint(path):
    path = Path(path).expanduser().absolute()
    if path.is_dir():
        candidates = [path / 'pytorch_model.pt', path / 'final_model/pytorch_model.pt']
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not path.is_file() or path.suffix != '.pt':
        raise FileNotFoundError(f'Expected a .pt checkpoint, final_model directory, or run directory: {path}')
    return path


def summarize(rows):
    metric_keys = sorted(set(rows[0]) - {'dataset', 'episode_id', 'start_frame', 'future_frame'})
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['dataset'], row['episode_id'])].append(row)
    episodes = []
    for (dataset, episode_id), values in sorted(grouped.items()):
        episodes.append({
            'dataset': dataset, 'episode_id': episode_id, 'windows': len(values),
            **{key: float(np.mean([row[key] for row in values])) for key in metric_keys},
        })
    window_mean = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    paired = {}
    for mode in ('teacher_latent', 'gt_future'):
        key = f'action_mse/{mode}/all'
        gains = np.asarray([ep['action_mse/predicted/all'] - ep[key] for ep in episodes])
        paired[mode] = {
            'episode_macro_mse_reduction': float(gains.mean()),
            'episodes_improved': int((gains > 0).sum()),
            'episodes_worsened': int((gains < 0).sum()),
        }
    return {
        'windows': len(rows), 'episodes': len(episodes), 'window_mean': window_mean,
        'episode_macro_mean': {key: float(np.mean([ep[key] for ep in episodes])) for key in metric_keys},
        'per_episode': episodes, 'paired_improvement': paired,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--dataset', help='One dataset from the saved training mixture; default: all.')
    parser.add_argument('--split', choices=('val', 'train'), default='val')
    parser.add_argument('--episodes', type=int, nargs='+', help='Restrict to IDs already in the selected split.')
    parser.add_argument('--windows-per-episode', type=int, default=64, help='Uniform full windows; 0 means all.')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-batches', type=int, help='Optional limit for a smoke test.')
    parser.add_argument('--noise-repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--num-inference-steps', type=int)
    parser.add_argument('--guidance-scale', type=float, default=1.0)
    parser.add_argument('--prefix-steps', type=int, help='Also measure the first N actions, e.g. actual replan length.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run-data', action='store_true', help='Validate data/collation without loading model weights.')
    args = parser.parse_args()
    for key in ('batch_size', 'noise_repeats'):
        if getattr(args, key) < 1:
            parser.error(f'--{key.replace("_", "-")} must be positive.')
    if args.num_workers < 0 or args.windows_per_episode < 0:
        parser.error('--num-workers and --windows-per-episode must be non-negative.')
    for key in ('max_batches', 'num_inference_steps', 'prefix_steps'):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            parser.error(f'--{key.replace("_", "-")} must be positive.')
    return args


def main():
    args = parse_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from starVLA.dataloader import _build_latent_world_collator
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from starVLA.model.framework.base_framework import baseframework
    from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
    from starVLA.model.framework.latent_world.runtime.offline_diagnostics import diagnose_train_batch
    from starVLA.model.framework.share_tools import read_mode_config

    checkpoint = resolve_checkpoint(args.checkpoint)
    raw_config, statistics = read_mode_config(checkpoint)
    cfg = OmegaConf.create(raw_config)
    policy_cfg = LatentWorldPolicyConfigBuilder(cfg).build()
    if str(policy_cfg.flow_future_feature) != 'predicted' or not policy_cfg.future_prediction:
        raise ValueError('Choose the normal WM checkpoint (future_prediction=true, flow_future_feature=predicted).')
    if args.prefix_steps is not None and args.prefix_steps > policy_cfg.action_horizon:
        raise ValueError('--prefix-steps exceeds the checkpoint action horizon.')
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.datasets.vla_data, resolve=True))
    if args.data_root is not None:
        data_cfg.data_root_dir = str(args.data_root)
    data_cfg.fingertip_sidecar_root = None
    data_cfg.enable_video_frame_cache = False
    data_cfg.enable_primary_video_aug = False
    data_cfg.enable_primary_random_resized_crop = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    mixture = get_vla_dataset(
        data_cfg, mode=args.split, framework_name='LaWAM',
        dataset_statistics_override=statistics, seed=args.seed,
    )
    windows = FixedEpisodeWindows(
        mixture, dataset_name=args.dataset, episode_ids=args.episodes,
        windows_per_episode=args.windows_per_episode,
    )
    collator = _build_latent_world_collator(
        cfg, policy_cfg=replace(policy_cfg, fingertip_loss_weight=0.0, geometry_conditioning=False, geometry_loss_weight=0.0), training=False,
    )
    loader = DataLoader(
        windows, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collator, shuffle=False, drop_last=False,
    )
    output_dir = args.output_dir.expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ('summary.json', 'per_window.csv', 'manifest.json'):
        if (output_dir / name).exists():
            raise FileExistsError(f'{output_dir / name} exists; choose a new --output-dir.')
    held_out = args.split == 'val' and not bool(data_cfg.get('train_split_all', True))
    manifest = {
        'checkpoint': str(checkpoint), 'checkpoint_run': str(checkpoint.parents[1]),
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'held_out_from_stage2': held_out,
        'action_horizon': policy_cfg.action_horizon,
        'episode_manifest': windows.episodes, 'skipped_short_episodes': windows.skipped_episodes,
        'windows': windows.windows,
        'notes': [
            'Oracle inputs use real future images. Offline diagnostic only, not deployable success rate.',
            'Action MSE/MAE are in checkpoint-normalized action units, without clipping.',
            'Each action comparison shares flow noise; reported errors average over noise repeats.',
            'Windows within an episode are correlated. More windows do not replace more held-out episodes.',
            'Teacher latent uses the checkpoint LAM encoder and the same video pair as distillation.',
            'GT future DINO bypasses the decoder and can be out of distribution for the action head.',
        ],
    }
    print(f'Planned: {len(windows)} windows from {len(windows.episodes)} episodes; held_out={held_out}', flush=True)
    if len(windows.episodes) < 5:
        print('WARNING: fewer than five episodes; interpret as a small-sample diagnostic.', flush=True)
    if not held_out:
        print('WARNING: this split is not confirmed held out from Stage 2.', flush=True)
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if args.dry_run_data:
        batch = next(iter(loader))
        shapes = {key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)}
        (output_dir / 'data_shapes.json').write_text(json.dumps(shapes, indent=2))
        print(json.dumps(shapes, indent=2))
        return

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run inside an allocated GPU job, or use --dry-run-data.')
    model = baseframework.from_pretrained(str(checkpoint)).to(device).eval()
    model.requires_grad_(False)
    backend = model.policy_backend
    groups = None
    if policy_cfg.flow_cfg.action_dim == 44 and all(
        mixture.datasets[window['dataset_index']].dataset_name.startswith('bimanual_') for window in windows.windows
    ):
        groups = {'arms': [*range(6), *range(22, 28)], 'hands': [*range(6, 22), *range(28, 44)]}
    rows = []
    offset = 0
    csv_stream = (output_dir / 'per_window.csv').open('w', newline='')
    try:
        writer = None
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            metrics = diagnose_train_batch(
                backend, batch, noise_repeats=args.noise_repeats,
                noise_seed=args.seed + batch_index * args.noise_repeats,
                num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                prefix_steps=args.prefix_steps, action_groups=groups,
            )
            for sample in range(batch['actions'].shape[0]):
                metadata = windows.windows[offset]
                row = {key: metadata[key] for key in ('dataset', 'episode_id', 'start_frame', 'future_frame')}
                row.update({key: float(value[sample]) for key, value in metrics.items()})
                if writer is None:
                    writer = csv.DictWriter(csv_stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
                offset += 1
            csv_stream.flush()
            print(f'Processed {offset}/{len(windows)} windows', flush=True)
    finally:
        csv_stream.close()
    report = summarize(rows)
    report['checkpoint'] = str(checkpoint)
    report['held_out_from_stage2'] = held_out
    report['notes'] = manifest['notes']
    (output_dir / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('\nMean normalized action MSE (lower is better):')
    for mode in ('predicted', 'teacher_latent', 'gt_future'):
        mean = report['window_mean'][f'action_mse/{mode}/all']
        print(f'  {mode:16s} {mean:.8f}')
    print(f'Report: {output_dir / "summary.json"}')


if __name__ == '__main__':
    main()
