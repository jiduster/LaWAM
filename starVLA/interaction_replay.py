"""Read raw Zarr v2 demos without depending on a specific zarr package version."""
import itertools
import json
from pathlib import Path

import numpy as np


def read_array(path):
    from numcodecs import get_codec
    path = Path(path)
    meta = json.loads((path/'.zarray').read_text())
    if meta['zarr_format'] != 2 or meta.get('filters'):
        raise ValueError(f'Only unfiltered Zarr v2 arrays supported: {path}')
    shape, chunks = tuple(meta['shape']), tuple(meta['chunks'])
    dtype = np.dtype(meta['dtype'])
    result = np.empty(shape, dtype=dtype)
    codec = get_codec(meta['compressor']) if meta['compressor'] else None
    for coord in itertools.product(*(range((s+c-1)//c) for s,c in zip(shape,chunks))):
        file = path/meta.get('dimension_separator','.').join(map(str,coord))
        encoded = file.read_bytes()  # Missing chunks fail; never synthesize archival data.
        decoded = codec.decode(encoded) if codec else encoded
        value = np.frombuffer(decoded,dtype=dtype).reshape(chunks,order=meta['order'])
        dst = tuple(slice(i*c,min((i+1)*c,s)) for i,c,s in zip(coord,chunks,shape))
        src = tuple(slice(0,d.stop-d.start) for d in dst)
        result[dst] = value[src]
    return result.reshape(shape[0],-1)


def match_source(recorded, converted_action, dataset):
    """Find the original static-prefix trim by exact state and rotvec-action correspondence."""
    matches=[]
    for ep,row in dataset.items():
        n=len(row['state'])
        trim=len(recorded)-n
        if trim < 0 or converted_action.shape != (len(recorded),44):
            continue
        if not np.allclose(recorded[trim,:46],row['state'][0],atol=1e-7,rtol=0):
            continue
        if (np.allclose(recorded[trim:,:46],row['state'],atol=1e-7,rtol=0)
                and np.allclose(converted_action[trim:],row['action'],atol=2e-7,rtol=0)):
            matches.append((ep,trim))
    return matches
