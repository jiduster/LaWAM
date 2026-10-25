"""Small, dependency-light readers for interaction experiment preflight/export."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def episode_metadata(root):
    files = sorted((Path(root) / 'meta/episodes').rglob('*.parquet'))
    if not files:
        raise FileNotFoundError(f'No LeRobot v3 episode metadata: {root}')
    rows = [row for file in files for row in pq.read_table(file).to_pylist()]
    ids = [int(row['episode_index']) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate episode metadata')
    return rows


def read_dataset(root):
    root = Path(root)
    files = sorted((root / 'data').rglob('*.parquet'))
    if not files:
        raise FileNotFoundError(root / 'data')
    columns = ['episode_index', 'frame_index', 'timestamp', 'observation.state', 'action']
    table = pa.concat_tables([pq.read_table(f, columns=columns) for f in files])
    ids = table['episode_index'].to_numpy()
    states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    frames = table['frame_index'].to_numpy()
    times = table['timestamp'].to_numpy()
    fps = float(json.loads((root / 'meta/info.json').read_text())['fps'])
    episodes = {}
    for meta in episode_metadata(root):
        ep = int(meta['episode_index'])
        selected = ids == ep
        n = int(meta['length'])
        if (selected.sum() != n or not np.array_equal(frames[selected], np.arange(n))
                or not np.allclose(times[selected], np.arange(n)/fps, atol=1e-4, rtol=0)):
            raise ValueError(f'Invalid episode frame/timestamp alignment: {ep}')
        state, action = states[selected], actions[selected]
        if state.shape != (n, 46) or action.shape != (n, 44) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f'Expected finite DexJoCo state/action [N,46/44]: {ep}')
        episodes[ep] = dict(state=state, action=action, timestamp=times[selected], frame_index=frames[selected], metadata=meta)
    return fps, episodes


def split_episodes(ids, ratio=.01):
    if len(ids) < 2:
        raise ValueError('Experiments require at least two episodes for train/validation')
    n = min(max(1, int(np.floor(len(ids)*ratio))), len(ids)-1)
    return ids[:-n], ids[-n:]


def audit_store(store, task, dataset, fps, ids):
    """Check every frame, not just episode/frame zero, before launching DDP."""
    for ep in ids:
        expected = dataset[ep]
        actual = store.episode(task, ep, fps, len(expected['state']))
        for name in ('state', 'action', 'timestamp'):
            if name not in actual or not np.allclose(actual[name], expected[name], atol=1e-5, rtol=0):
                raise ValueError(f'Geometry/dataset {name} mismatch in episode {ep}')


def feature_statistics(store, task, fps, training):
    from starVLA.interaction_geometry import FEATURE_DIM
    total = np.zeros(FEATURE_DIM)
    square = np.zeros(FEATURE_DIM)
    count = np.zeros(FEATURE_DIM)
    for ep in training:
        z = store.episode(task, ep, fps)
        valid = z['interaction_valid']
        values = np.where(valid, z['interaction_geometry'], 0).astype(np.float64)
        total += values.sum((0, 1))
        square += (values**2).sum((0, 1))
        count += valid.sum((0, 1))
    mean = total / np.maximum(count, 1)
    std = np.maximum(np.sqrt(np.maximum(square/np.maximum(count, 1)-mean**2, 0)), 1e-3)
    # Structurally absent features (e.g. Hanoi joint angle) remain neutral.
    std[count == 0] = 1
    return dict(mean=mean.tolist(), std=std.tolist(), valid_feature_counts=count.astype(int).tolist())
