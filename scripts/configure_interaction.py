"""Resolve portable assets, audit a common cohort, and generate matched Stage 2 configs."""
import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.interaction_data import read_dataset, split_episodes, audit_store, feature_statistics
from starVLA.interaction_geometry import ENTITY_NAMES, FEATURE_DIM, InteractionStore

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / 'starVLA/config/training/starvla_train_dexjoco_interaction_stage2.yaml'
VARIANTS = ('baseline', 'oracle', 'visual', 'visual_aux')
WM_VARIANTS = ('visual_wm_aux', 'visual_wm_aux_detach')
TASKS = {'hanoi': 'bimanual_hanoi', 'microwave': 'bimanual_microwave_cook'}


def configure(args):
    task = TASKS.get(args.task, args.task)
    data_root = args.data_root.expanduser().resolve()
    dataset_root = data_root if data_root.name == task else data_root / task
    fps, dataset = read_dataset(dataset_root)
    if fps != 30:
        raise ValueError('H30 one-second recipe requires 30 FPS')
    ids = list(dataset)
    if args.episodes:
        selected = json.loads(args.episodes.read_text())
        if not isinstance(selected, list) or len(selected) != len(set(selected)) or set(selected)-set(ids):
            raise ValueError('--episodes must be a JSON list of known, unique episode IDs')
        ids = [ep for ep in ids if ep in set(selected)]
    training, validation = split_episodes(ids)
    variants = VARIANTS if args.variant == 'all' else (args.variant,)
    stats = None
    acceptance = None
    if args.sidecar_root is not None:
        store = InteractionStore(args.sidecar_root)
        manifest = store.manifest(task)
        accepted_ids = manifest.get('accepted_episodes')
        if accepted_ids is not None and set(ids) != set(accepted_ids):
            raise ValueError('Use the accepted_episodes.json cohort for all four variants')
        audit_store(store, task, dataset, fps, ids)
        acceptance = {k: manifest.get(k) for k in (
            "label_source", "acceptance_policy", "acceptance_note", "source_manifest_sha256")}
    if any(v in ('oracle', 'visual_aux', *WM_VARIANTS) for v in variants):
        if args.sidecar_root is None:
            raise ValueError('Oracle/Visual Aux need --sidecar-root with approved current-frame geometry')
        store = InteractionStore(args.sidecar_root)
        audit_store(store, task, dataset, fps, ids)
        stats = feature_statistics(store, task, fps, training)
    asset = args.asset_root.expanduser().resolve()
    cfg = OmegaConf.load(TEMPLATE)
    am = cfg.framework.action_model
    paths = {
        'vlm': args.vlm or asset/'results/Checkpoints/qwen3_weights',
        'dino': args.dino or asset/'results/Checkpoints/dinov3-vitb16-pretrain-lvd1689m',
        'lam': args.lam or asset/'latent_action_model/logs/dino_large_vae_dexjoco_bimanual_exclude_photograph_fingertip_stage1/pytorch_model.pt',
        'pretrain': args.pretrain or asset/'results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt',
        'lam_yaml': args.lam_yaml or asset/'latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml',
    }
    paths = {k: Path(p).expanduser().resolve() for k, p in paths.items()}
    for name, path in paths.items():
        if not path.exists() or (name in ('vlm', 'dino') and not (path/'config.json').is_file()):
            raise FileNotFoundError(f'Missing {name} asset: {path}')
    if args.steps <= 0:
        raise ValueError('steps must be positive')
    cfg.datasets.vla_data.data_root_dir = str(dataset_root.parent)
    cfg.datasets.vla_data.data_mix = 'dexjoco_'+task
    cfg.datasets.vla_data.episode_indices = ids
    cfg.framework.qwenvl.base_vlm = str(paths['vlm'])
    am.lam_ckpt_path = str(paths['lam'])
    cfg.trainer.pretrained_checkpoint = str(paths['pretrain'])
    cfg.trainer.max_train_steps = args.steps
    cfg.trainer.num_warmup_steps = min(1000, max(1, args.steps//10))
    cfg.trainer.save_interval = args.save_interval
    if args.save_interval < 1:
        raise ValueError('save-interval must be positive')
    cfg.seed = args.seed
    if args.fingertip_root:
        # Optional existing endpoint supervision must be identical for all variants.
        import numpy as np
        tip_root = args.fingertip_root.expanduser().resolve()
        tip_root = tip_root if tip_root.name == task else tip_root/task
        for ep in ids:
            with np.load(tip_root/'episodes'/f'episode-{ep:06d}.npz', allow_pickle=False) as z:
                n = len(dataset[ep]['state'])
                if (z['tip_position'].shape != (n, 8, 3) or z['valid_mask'].shape != (n, 8)
                        or not np.isfinite(z['tip_position'][z['valid_mask'].astype(bool)]).all()):
                    raise ValueError(f'Invalid endpoint sidecar for episode {ep}')
        cfg.datasets.vla_data.fingertip_sidecar_root = str(args.fingertip_root.resolve())
        am.fingertip_loss_weight = .1
        am.fingertip_target_mode = 'endpoint'
        am.fingertip_num_points = 8
        am.fingertip_target_scale = .1
    out = args.config_root.expanduser().resolve()/task
    out.mkdir(parents=True, exist_ok=True)
    lam = OmegaConf.load(paths['lam_yaml'])
    lam.model.vision_model_id = str(paths['dino'])
    lam_yaml = out/'lam.yaml'
    OmegaConf.save(lam, lam_yaml)
    am.lam_yaml_path = str(lam_yaml)
    cohort = dict(task=task, episodes=ids, training_episodes=training, validation_episodes=validation,
                  data_state_sha256=hashlib.sha256(b''.join(dataset[ep]['state'].tobytes()+dataset[ep]['action'].tobytes() for ep in ids)).hexdigest())
    if args.sidecar_root is not None:
        manifest_path = InteractionStore(args.sidecar_root).dataset_root(task)/'manifest.json'
        cohort['geometry_manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    cohort['sha256'] = hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest()
    cfg.interaction_experiment = dict(cohort, assets={k:str(v) for k,v in paths.items()},
                                      geometry_statistics=stats, geometry_acceptance=acceptance)
    files = []
    for variant in variants:
        current = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        policy = current.framework.action_model
        wm_mode = variant in WM_VARIANTS
        enabled, labels = variant != 'baseline', variant in ('oracle', 'visual_aux', *WM_VARIANTS)
        policy.interaction_mode = 'none' if not enabled else variant
        policy.interaction_task = task
        policy.interaction_entities = len(ENTITY_NAMES[task])
        policy.interaction_feature_dim = FEATURE_DIM
        policy.interaction_loss_weight = args.aux_weight if variant in ('visual_aux', *WM_VARIANTS) else 0
        policy.flow_cfg.interaction_conditioning = enabled and not wm_mode
        if wm_mode:
            # This auxiliary branch has no route to the action expert except via future DINO.
            from starVLA.model.framework.vlas.wm_geometry import dynamic_geometry_mask
            current.interaction_experiment.future_geometry = dict(
                timestamp='last sampled WM video frame; padded endpoints masked',
                supervision_mask=dynamic_geometry_mask(task).tolist(),
                detach=variant == 'visual_wm_aux_detach', weight=args.aux_weight)
        if labels:
            policy.interaction_sidecar_root = str(args.sidecar_root.expanduser().resolve())
            policy.interaction_feature_mean = stats['mean']
            policy.interaction_feature_std = stats['std']
            # Estimated-pose visual_aux may have fully masked frames. Oracle stays strict.
            sidecar_manifest = InteractionStore(args.sidecar_root).manifest(task)
            policy.interaction_allow_empty_frames = (
                bool(sidecar_manifest.get('allow_empty_frames', False)) and variant != 'oracle')
        current.datasets.vla_data.load_interaction_geometry = labels
        current.datasets.vla_data.load_future_interaction_geometry = wm_mode
        current.datasets.vla_data.interaction_sidecar_root = str(args.sidecar_root.expanduser().resolve()) if labels else ''
        current.run_id = f'{task}_interaction_{variant}_h30_{args.steps}_seed{args.seed}'
        current.run_root_dir = str(args.output_root.expanduser().resolve()/task/variant)
        from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
        built = LatentWorldPolicyConfigBuilder(current).build()
        assert built.action_horizon == 30 and not built.action_expert_only
        file = out/f'{variant}.yaml'
        OmegaConf.save(current, file)
        files.append(file)
        print(f'{variant}: {file}')
    (out/'cohort.json').write_text(json.dumps(cohort, indent=2)+'\n')
    print(f'Train/val={len(training)}/{len(validation)}, H=30, 8 GPUs x batch 4 = 32, steps={args.steps}')
    print(f'Cohort={cohort["sha256"]}; compare only configurations with the same cohort and assets.')
    return files


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True, choices=tuple(TASKS)+tuple(ENTITY_NAMES))
    p.add_argument('--variant', default='all', choices=('all',)+VARIANTS+WM_VARIANTS)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--sidecar-root', type=Path)
    p.add_argument('--episodes', type=Path, help='Same explicit JSON episode list for all four variants; default all episodes')
    p.add_argument('--asset-root', type=Path, default=REPO)
    for name in ('vlm', 'dino', 'lam', 'pretrain', 'lam-yaml'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--fingertip-root', type=Path, help='Optional matched endpoint supervision (off by default for all four)')
    p.add_argument('--config-root', type=Path, default=REPO/'results/interaction_configs')
    p.add_argument('--output-root', type=Path, default=REPO/'results/Checkpoints/interaction')
    p.add_argument('--steps', type=int, default=40000)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--save-interval', type=int, default=10000)
    p.add_argument('--aux-weight', type=float, default=.1)
    return p


if __name__ == '__main__':
    configure(parser().parse_args())
