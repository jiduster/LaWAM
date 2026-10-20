"""Validate current simulator recordings against LeRobot before publishing sidecars."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from starVLA.contact_geometry import (
    SCHEMA, FINGER_ORDER, ContactStore, parquet_fingerprints, validate_geometry,
)


def prepare(dataset_root, output_root, recordings=None, validate_only=False):
    dataset_root, output_root = Path(dataset_root), Path(output_root)
    dataset = dataset_root.name
    if dataset != "bimanual_photograph":
        raise ValueError("This schema currently supports bimanual_photograph only")
    with (dataset_root / "meta/info.json").open() as stream:
        fps = float(json.load(stream)["fps"])
    hashes = parquet_fingerprints(dataset_root)
    if validate_only:
        store = ContactStore(output_root)
        meta = store.manifest(dataset)
        if hashes != meta["parquet_sha256"]:
            raise ValueError("LeRobot parquet changed since contact alignment validation")
        for episode, count in meta["episodes"].items():
            store.frame(dataset, int(episode), count - 1, fps)
        print(f"Validated {len(meta['episodes'])} contact episodes against LeRobot parquet hashes")
        return
    if recordings is None:
        raise ValueError("--recordings must contain synchronized simulator recordings; reset poses are insufficient")
    columns = ["episode_index", "frame_index", "timestamp", "observation.state"]
    import pandas as pd
    table = pd.concat([pq.read_table(dataset_root / path, columns=columns).to_pandas()
                       for path in hashes], ignore_index=True)
    episodes = {}
    validated = []
    for episode, rows in table.groupby("episode_index", sort=True):
        episode = int(episode)
        rows = rows.sort_values("frame_index")
        path = Path(recordings) / f"episode-{episode:06d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema"].item()) != SCHEMA:
                raise ValueError(f"Recording schema mismatch: {path}")
            geometry = validate_geometry(archive["contact_geometry"]).copy()
            if geometry.shape != (len(rows), 9, 6):
                raise ValueError(f"Frame count mismatch: {path}")
            indices = rows["frame_index"].to_numpy()
            if not np.array_equal(indices, np.arange(len(rows))) or not np.array_equal(archive["frame_index"], indices):
                raise ValueError(f"Frame index mismatch: {path}")
            state = np.stack(rows["observation.state"])
            if archive["state"].shape != state.shape or not np.allclose(archive["state"], state, atol=1e-5, rtol=0):
                raise ValueError(f"Raw robot state mismatch (wrong episode or timing): {path}")
            if archive["timestamp"].shape != (len(rows),) or not np.allclose(archive["timestamp"], rows["timestamp"].to_numpy(), atol=1e-5, rtol=0):
                raise ValueError(f"Timestamp mismatch: {path}")
        episodes[str(episode)] = len(rows)
        validated.append((episode, geometry))
    target = output_root / dataset
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite sidecars: {target}")
    (target / "episodes").mkdir(parents=True)
    for episode, geometry in validated:
        np.savez_compressed(target / "episodes" / f"episode-{episode:06d}.npz",
                            contact_geometry=geometry, frame_index=np.arange(len(geometry)))
    with (target / "manifest.json").open("x") as stream:
        json.dump(dict(schema=SCHEMA, dataset=dataset, fps=fps, finger_order=FINGER_ORDER,
                       episodes=episodes, parquet_sha256=hashes), stream, indent=2)
    print(f"Prepared {len(episodes)} aligned oracle contact episodes at {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--recordings", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    prepare(**vars(parser.parse_args()))
