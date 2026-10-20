"""Shared CPU-only oracle data contract for recording, training and serving."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "photograph_shutter_world_v1"
FINGER_ORDER = [f"{finger}_tip_{side}" for side in ("left", "right")
                for finger in ("ff", "mf", "rf", "th")]


def validate_geometry(value):
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-2:] != (9, 6) or not np.isfinite(value).all():
        raise ValueError("Expected finite contact geometry [...,9,6]")
    if not np.allclose(np.linalg.norm(value[..., 0, 3:], axis=-1), 1, atol=1e-3):
        raise ValueError("Shutter normal must be unit length")
    if np.any(value[..., 1:, 3:] != 0):
        raise ValueError("Fingertip normal slots must be zero")
    return value


def extract_photograph_contact(model, data):
    """Current geometry after mj_forward/step; no reset pose or future state.

    Target = outward face centre of shutter cylinder. In camera.xml its -Z
    points out of the camera body (+Y in the camera body frame).
    Tips = centres of collision capsules, NOT distal joint origins and NOT
    an assertion that these points are touching the shutter.
    """
    import mujoco

    target_id = model.geom("camera_body_top_cap").id
    if model.geom_type[target_id] != mujoco.mjtGeom.mjGEOM_CYLINDER:
        raise ValueError("Expected cylinder camera_body_top_cap")
    normal = -data.geom_xmat[target_id].reshape(3, 3)[:, 2]
    result = np.zeros((9, 6), dtype=np.float32)
    result[0, :3] = data.geom_xpos[target_id] + normal * model.geom_size[target_id, 1]
    result[0, 3:] = normal
    for i, name in enumerate(FINGER_ORDER, start=1):
        body_id = model.body(name).id
        candidates = np.flatnonzero(
            (model.geom_bodyid == body_id)
            & (model.geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE)
            & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        )
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one fingertip collision capsule on {name}")
        result[i, :3] = data.geom_xpos[candidates[0]]
    return validate_geometry(result)


def parquet_fingerprints(dataset_root):
    root = Path(dataset_root)
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet data in {root}")
    result = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[str(path.relative_to(root))] = digest.hexdigest()
    return result


class ContactStore:
    def __init__(self, root):
        if not str(root).strip():
            raise ValueError("contact_sidecar_root is required for contact training")
        self.root = Path(root)
        self.cache = OrderedDict()
        self.manifests = {}

    def manifest(self, dataset):
        if Path(dataset).name != dataset:
            raise ValueError("dataset must be a directory basename")
        if dataset not in self.manifests:
            with (self.root / dataset / "manifest.json").open() as stream:
                meta = json.load(stream)
            if meta["schema"] != SCHEMA or meta["dataset"] != dataset or meta["finger_order"] != FINGER_ORDER:
                raise ValueError("Contact sidecar schema/dataset/finger order mismatch")
            self.manifests[dataset] = meta
        return self.manifests[dataset]

    def frame(self, dataset, episode, frame, fps):
        meta = self.manifest(dataset)
        if not np.isclose(float(meta["fps"]), fps):
            raise ValueError("Contact sidecar FPS mismatch")
        key = (dataset, int(episode))
        if key not in self.cache:
            path = self.root / dataset / "episodes" / f"episode-{episode:06d}.npz"
            with np.load(path, allow_pickle=False) as archive:
                geometry = validate_geometry(archive["contact_geometry"]).copy()
                indices = archive["frame_index"]
                if geometry.shape != (meta["episodes"][str(episode)], 9, 6):
                    raise ValueError(f"Contact sidecar length mismatch: {path}")
                if not np.array_equal(indices, np.arange(len(geometry))):
                    raise ValueError(f"Non-contiguous contact frame indices: {path}")
            self.cache[key] = geometry
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        if not 0 <= frame < len(self.cache[key]):
            raise IndexError(f"Contact frame out of range: {key}, {frame}")
        return self.cache[key][frame].copy()


class PhotographContactRecorder:
    """Call append at the SAME instant as each demonstration observation.

    state must be the raw 46D LeRobot state, before normalization; timestamp
    and frame_index must match the saved image/state row. One recorder/episode.
    """

    def __init__(self):
        self.rows = []

    def append(self, model, data, state, frame_index, timestamp):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (46,) or not np.isfinite(state).all():
            raise ValueError("Recorder requires finite raw state[46]")
        if frame_index != len(self.rows):
            raise ValueError("Recorder requires contiguous frame indices starting at zero")
        self.rows.append((extract_photograph_contact(model, data), state.copy(), float(timestamp)))

    def save(self, path):
        if not self.rows:
            raise ValueError("Cannot save empty contact recording")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids silently replacing a different demonstration.
        with path.open("xb") as stream:
            np.savez_compressed(stream, schema=SCHEMA,
                                contact_geometry=np.stack([r[0] for r in self.rows]),
                                state=np.stack([r[1] for r in self.rows]),
                                timestamp=np.array([r[2] for r in self.rows]),
                                frame_index=np.arange(len(self.rows)))
