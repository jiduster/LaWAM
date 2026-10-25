"""Validate synchronous simulator recordings against LeRobot and publish a sidecar.

Use for newly captured demonstrations with InteractionRecorder. This deliberately
rejects replay candidates: use the separate replay audit/review workflow for them.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from starVLA.interaction_data import read_dataset
from starVLA.interaction_geometry import SCHEMA, ENTITY_NAMES, FEATURE_NAMES, validate_geometry


def run(args):
    task=args.task
    data=args.data_root if args.data_root.name==task else args.data_root/task
    source=args.recording_root if args.recording_root.name==task else args.recording_root/task
    fps,rows=read_dataset(data)
    target=args.output/task
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite sidecar directory: {target}')
    selected=list(rows) if args.episodes is None else json.loads(args.episodes.read_text())
    if not selected or len(selected)!=len(set(selected)) or set(selected)-set(rows):
        raise ValueError('Invalid episode list')
    entries={}
    files=[]
    for ep in selected:
        file=source/f'episode-{ep:06d}.npz'
        with np.load(file,allow_pickle=False) as z:
            for key,expected in [('schema',SCHEMA),('task',task),('label_source','synchronous_simulator_capture')]:
                if z[key].item()!=expected:
                    raise ValueError(f'Synchronous recording provenance mismatch: {file}, {key}')
            value,valid=z['interaction_geometry'],z['interaction_valid']
            validate_geometry(value,valid,task)
            n=len(rows[ep]['state'])
            if value.shape!=(n,len(ENTITY_NAMES[task]),len(FEATURE_NAMES)):
                raise ValueError(f'Geometry length mismatch: {file}')
            for key in ('frame_index','timestamp','state','action'):
                if z[key].shape!=rows[ep][key].shape or not np.allclose(z[key],rows[ep][key],atol=1e-5,rtol=0):
                    raise ValueError(f'Recording/LeRobot {key} alignment mismatch: {file}')
            entries[str(ep)]=dict(length=n,valid=True,source=str(file))
            files.append(file)
    (target/'episodes').mkdir(parents=True)
    for file in files:
        shutil.copyfile(file,target/'episodes'/file.name)
    manifest=dict(schema=SCHEMA,task=task,features=FEATURE_NAMES,entities=ENTITY_NAMES[task],
                  coordinate_frame='world',fps=fps,episodes=entries,label_source='synchronous_simulator_capture')
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (target/'approved_episodes.json').write_text(json.dumps(selected,indent=2)+'\n')
    print(target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=tuple(ENTITY_NAMES),required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--recording-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=Path)
    run(p.parse_args())
