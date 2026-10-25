"""Reconstruct candidate CURRENT multi-object labels from raw demonstrations.

Robot correspondence is necessary but cannot certify object trajectory alignment.
A candidate manifest remains invalid until explicitly reviewed against original RGB.
This tool leaves all source demos and training Parquet unchanged.
"""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from starVLA.interaction_data import read_dataset
from starVLA.interaction_geometry import SCHEMA, ENTITY_NAMES, FEATURE_NAMES, extract_interaction_geometry
from starVLA.interaction_replay import read_array, match_source

TASKS = {'hanoi':'bimanual_hanoi','microwave':'bimanual_microwave_cook'}
PROPRIO = {
    'bimanual_hanoi': ('tcp_pose','gripper_pose','hanoi_base_ori_pos','table_delta_height'),
    'bimanual_microwave_cook': ('tcp_pose','gripper_pose','hot_dog_ori_pose','microwave_ori_pose','table_delta_height'),
}
CLASSES = {'bimanual_hanoi':'PandaBimanualHanoiGymEnv','bimanual_microwave_cook':'PandaBimanualMicrowaveCookGymEnv'}


class PhysicsOnlyViewer:
    def __init__(self,*args,**kwargs): self._viewers={}
    def render(self,*args,**kwargs): return None
    def close(self): pass


def run(args):
    import mujoco
    from dexjoco.tasks.state_restorers import _split_state_by_proprio, _STATE_RESTORERS
    task=TASKS.get(args.task,args.task)
    root=args.data_root if args.data_root.name==task else args.data_root/task
    raw_root=args.raw_root if args.raw_root.name==task else args.raw_root/task
    fps,dataset=read_dataset(root)
    wanted=set(dataset) if args.episodes is None else set(args.episodes)
    if wanted-set(dataset): raise ValueError('Unknown episode requested')
    output=args.output/task
    if (output/'manifest.json').exists():
        raise FileExistsError(f'Refusing to overwrite an export: {output}; use another --output')
    (output/'episodes').mkdir(parents=True,exist_ok=True)
    module=importlib.import_module('dexjoco.sim.envs.panda_'+task+'_env')
    # Keep pacing: Hanoi illegal-placement recovery uses wall-clock deadlines.
    # Only disable display rendering, leaving controllers and task transitions intact.
    module.MujocoRenderer=PhysicsOnlyViewer
    env_cls=getattr(module,CLASSES[task])
    try:
        revision=subprocess.check_output(['git','-C',str(Path(importlib.import_module('dexjoco').__file__).parent),'rev-parse','HEAD'],text=True).strip()
    except (subprocess.CalledProcessError,TypeError): revision='unknown'
    manifest=dict(schema=SCHEMA,task=task,entities=ENTITY_NAMES[task],features=FEATURE_NAMES,
                  coordinate_frame='world',fps=fps,episodes={},label_source='mujoco_raw_replay_reconstruction',
                  caveat='Candidate reconstructed labels, not independently measured original object poses; RGB review required.',
                  mujoco_version=mujoco.__version__,dexjoco_revision=revision,seed=args.seed)
    used=set()
    for source in sorted(raw_root.glob('*/replay.zarr')):
        recorded=read_array(source/'data/state').astype(np.float32)
        converted=read_array(source/'data/action_rotvec').astype(np.float32)
        matches=[(ep,trim) for ep,trim in match_source(recorded,converted,dataset) if ep in wanted]
        if not matches: continue
        if len(matches)!=1: raise ValueError(f'Ambiguous source mapping {source}: {matches}')
        ep,trim=matches[0]
        if ep in used: raise ValueError(f'Multiple raw sources match episode {ep}; deduplicate sources explicitly')
        used.add(ep)
        actions=read_array(source/'data/action')
        if actions.shape!=(len(recorded),46): raise ValueError(f'Raw actions must be [N,46]: {source}')
        print(f'episode {ep}: {source.parent.name}, trim={trim}, steps={len(actions)}',flush=True)
        env=env_cls(seed=args.seed,randomize=False,randomize_dynamics=False,image_obs=False)
        env._prime_rgb_array_renderer=lambda:None
        renderer=None
        video=None
        try:
            env.reset()
            reference=env._compute_observation()['state']
            parts=_split_state_by_proprio(recorded[0],PROPRIO[task],reference)
            _STATE_RESTORERS[task](env,parts)
            mujoco.mj_forward(env._model,env._data)
            geometries,masks,states,qposes=[],[],[],[]
            if args.preview:
                import imageio.v2 as imageio
                renderer=mujoco.Renderer(env._model,height=640,width=640)
                video=imageio.get_writer(str(output/f'episode-{ep:06d}-replay.mp4'),fps=fps)
            for frame,action in enumerate(actions):
                current=env._compute_observation()['state']
                states.append(np.concatenate([np.asarray(current[k]).ravel() for k in PROPRIO[task]])[:46])
                if frame>=trim:
                    geometry,mask=extract_interaction_geometry(env._model,env._data,task)
                    geometries.append(geometry)
                    masks.append(mask)
                    qposes.append(env._data.qpos.copy())
                    if renderer is not None:
                        renderer.update_scene(env._data,camera=int(env._front_camera_id))
                        video.append_data(renderer.render())
                env.step({'right':action[:23].astype(np.float64),'left':action[23:].astype(np.float64)})
            observed=np.asarray(states)[trim:].astype(np.float32)
            row=dataset[ep]
            error=float(np.max(np.abs(observed-row['state'])))
            numeric=bool(error<1e-5)
            path=output/'episodes'/f'episode-{ep:06d}.npz'
            np.savez_compressed(path,interaction_geometry=np.stack(geometries),interaction_valid=np.stack(masks),
                                state=row['state'],action=row['action'],replay_state=observed,
                                timestamp=row['timestamp'],frame_index=row['frame_index'],
                                source_raw_frame_index=np.arange(trim,len(actions)),qpos=np.stack(qposes))
            # No task success gate: selecting successful replays would bias the data cohort.
            manifest['episodes'][str(ep)]=dict(length=len(observed),valid=False,numerical_alignment_pass=numeric,
                                               replay_state_max_abs=error,raw_source=str(source),trim=trim,
                                               visual_review_pass=False)
            print(f'episode {ep}: state max abs={error:.3g}; candidate only, numeric_pass={numeric}',flush=True)
        finally:
            if video is not None: video.close()
            if renderer is not None: renderer.close()
            env.close()
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    missing=sorted(wanted-used)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'missing_episodes.json').write_text(json.dumps(missing)+'\n')
    print(f'Exported {len(used)}, missing={missing}. Review original RGB against replay before approval.')
    if missing: raise RuntimeError('Incomplete raw sources: see missing_episodes.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True,choices=tuple(TASKS)+tuple(ENTITY_NAMES))
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,nargs='+')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--preview',action='store_true',help='Render replay ego video at original timestamps for visual audit')
    run(p.parse_args())
