"""Record an explicit visual review of candidate reconstructions; reject numeric failures."""
import argparse
import json
from pathlib import Path


def approve(root, episodes, note):
    if not note.strip(): raise ValueError('A review note is required')
    manifest=root/'manifest.json'
    meta=json.loads(manifest.read_text())
    ids=json.loads(episodes.read_text())
    if not isinstance(ids,list) or len(ids)!=len(set(ids)) or not ids:
        raise ValueError('Expected nonempty JSON list of reviewed episode IDs')
    for ep in ids:
        entry=meta['episodes'][str(ep)]
        if not entry['numerical_alignment_pass'] or entry['replay_state_max_abs']>=1e-5:
            raise ValueError(f'Cannot approve episode {ep}: numeric correspondence failed')
        if not (root/f'episode-{ep:06d}-replay.mp4').is_file():
            raise FileNotFoundError(f'No replay preview for reviewed episode {ep}; re-export with --preview')
    for ep,entry in meta['episodes'].items():
        entry['visual_review_pass']=int(ep) in set(ids)
        entry['valid']=bool(entry['numerical_alignment_pass'] and entry['visual_review_pass'])
    meta['visual_review_note']=note
    manifest.write_text(json.dumps(meta,indent=2)+'\n')
    (root/'approved_episodes.json').write_text(json.dumps(ids,indent=2)+'\n')
    print(root/'approved_episodes.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True,help='Task sidecar directory')
    p.add_argument('--episodes',type=Path,required=True,help='JSON list YOU reviewed against original RGB')
    p.add_argument('--note',required=True)
    approve(**vars(p.parse_args()))
