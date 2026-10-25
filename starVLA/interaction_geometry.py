"""Current-frame multi-object geometry, shared by export, training and online inference.

World coordinates (meters), fixed semantic slots. Never reconstruct dynamic objects
from robot FK or initial object poses. See docs/multi_object_interaction.md.
"""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np


SCHEMA = "dexjoco_interaction_world_v2"
FEATURE_DIM = 17
FEATURE_NAMES = [f"position_{a}" for a in "xyz"] + [f"rotation_{col}_{a}" for col in ("x", "y") for a in "xyz"] + [
    f"relative_to_{hand}_{a}" for hand in ("left", "right") for a in "xyz"
] + ["joint_angle", "extent"]
ENTITY_NAMES = {
    "bimanual_hanoi": ["base", "disk_large", "disk_medium", "disk_small", "post_a", "post_b", "post_c"],
    "bimanual_microwave_cook": ["hot_dog", "microwave_body", "door", "door_handle", "start_button", "interior"],
}


def validate_geometry(value, valid, task):
    value, valid = np.asarray(value), np.asarray(valid)
    expected = (len(ENTITY_NAMES[task]), FEATURE_DIM)
    if value.shape[-2:] != expected or valid.shape != value.shape or valid.dtype != np.bool_:
        raise ValueError(f"Expected interaction geometry/mask [...,{expected}], got {value.shape}/{valid.shape}")
    if not np.isfinite(value[valid]).all() or not valid.any(axis=(-2, -1)).all():
        raise ValueError("Interaction geometry needs finite valid features in every frame")
    for start in (3, 6):
        axis_valid = valid[..., start:start+3].all(-1)
        if not np.allclose(np.linalg.norm(value[..., start:start+3], axis=-1)[axis_valid], 1, atol=1e-4):
            raise ValueError("Interaction rotation columns must have unit length")
    rotation_valid = valid[..., 3:9].all(-1)
    dot = np.sum(value[..., 3:6] * value[..., 6:9], axis=-1)
    if not np.allclose(dot[rotation_valid], 0, atol=1e-4):
        raise ValueError("Interaction rotation columns must be orthogonal")


def extract_interaction_geometry(model, data, task):
    """Call AFTER mj_forward, in the same simulator thread as RGB/state capture.

    Rotation = the first two columns of the world rotation matrix (6D).
    Only door/handle joint features are valid.
    Extent = bounding radius for disks, radius for posts, bounding half-extent
    for boxes/interior. Body-only entries without a defined extent are masked.
    """
    left = data.sensor("franka/flange_pos_left").data.copy()
    right = data.sensor("franka/flange_pos_right").data.copy()
    values, masks = [], []

    def add(position, rotation, angle=None, extent=None):
        rotation = np.asarray(rotation).reshape(3, 3)
        value = np.r_[position, rotation[:, 0], rotation[:, 1], position-left, position-right,
                      0.0 if angle is None else angle, 0.0 if extent is None else extent]
        valid = np.ones(FEATURE_DIM, dtype=bool)
        valid[15], valid[16] = angle is not None, extent is not None
        values.append(value)
        masks.append(valid)

    if task == "bimanual_hanoi":
        base = data.body("hanoi_base")
        add(base.xpos.copy(), base.xmat)
        for size in ("large", "medium", "small"):
            body = data.body(f"hanoi_disk_{size}")
            # Outer radius from the actual box collision geoms, not hardcoded reset positions.
            radius = max(np.max(np.abs(model.geom(f"hanoi_disk_{size}_collision_{side}").pos[:2])
                                + model.geom(f"hanoi_disk_{size}_collision_{side}").size[:2])
                         for side in ("pos_x", "neg_x", "pos_y", "neg_y"))
            add(body.xpos.copy(), body.xmat, extent=radius)
        for post in "abc":
            geom = data.geom(f"hanoi_post_{post}_collision")
            add(geom.xpos.copy(), geom.xmat, extent=float(model.geom(f"hanoi_post_{post}_collision").size[0]))
    elif task == "bimanual_microwave_cook":
        hotdog = data.body("hot_dog")
        add(hotdog.xpos.copy(), hotdog.xmat)
        body, door = data.body("microwave_object"), data.body("door")
        angle = float(data.jnt("microjoint").qpos[0])
        add(body.xpos.copy(), body.xmat)
        # door_p1 is on the moving door surface; body origin lies near the hinge.
        add(data.site("door_p1").xpos.copy(), door.xmat, angle=angle)
        for name in ("door_handle", "start_button"):
            geom = data.geom(name)
            add(geom.xpos.copy(), geom.xmat, angle=angle if name == "door_handle" else None,
                extent=float(np.max(model.geom(name).size)))
        origin = data.site("int_p0").xpos.copy()
        edges = [data.site("int_p"+a).xpos.copy()-origin for a in "xyz"]
        add(origin+sum(edges)/2, body.xmat, extent=max(np.linalg.norm(edge)/2 for edge in edges))
    else:
        raise ValueError(f"Unsupported interaction task {task}")
    value, valid = np.asarray(values, dtype=np.float32), np.asarray(masks, dtype=bool)
    validate_geometry(value, valid, task)
    return value, valid


class InteractionStore:
    """Validate sidecar schema and complete episode alignment; bounded worker cache."""
    def __init__(self, root, cache_size=4):
        self.root = Path(root).expanduser()
        self.manifests = {}
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def dataset_root(self, task):
        return self.root if self.root.name == task else self.root / task

    def manifest(self, task):
        if task not in self.manifests:
            meta = json.loads((self.dataset_root(task)/"manifest.json").read_text())
            if (meta.get("schema") != SCHEMA or meta.get("task") != task
                    or meta.get("entities") != ENTITY_NAMES[task]
                    or meta.get("features") != FEATURE_NAMES or meta.get("coordinate_frame") != "world"):
                raise ValueError(f"Interaction sidecar schema mismatch for {task}")
            self.manifests[task] = meta
        return self.manifests[task]

    def episode(self, task, episode, fps, length=None):
        meta = self.manifest(task)
        if not np.isclose(float(meta["fps"]), fps, atol=1e-6):
            raise ValueError("Interaction sidecar FPS mismatch")
        key = (task, int(episode))
        if key not in self.cache:
            entry = meta["episodes"].get(str(int(episode)))
            if entry is None or not entry.get("valid", False):
                raise ValueError(f"Missing/invalid geometry episode {key}; use the same audited episode subset for all four variants")
            path = self.dataset_root(task)/"episodes"/f"episode-{episode:06d}.npz"
            with np.load(path, allow_pickle=False) as z:
                values = {name: z[name].copy() for name in (
                    "interaction_geometry", "interaction_valid", "frame_index", "timestamp", "state", "action")}
            n = int(entry["length"])
            validate_geometry(values["interaction_geometry"], values["interaction_valid"], task)
            if (values["interaction_geometry"].shape != (n, len(ENTITY_NAMES[task]), FEATURE_DIM)
                    or values["state"].shape != (n, 46)
                    or values["action"].shape != (n, 44)
                    or not np.isfinite(values["state"]).all()
                    or not np.isfinite(values["action"]).all()
                    or not np.array_equal(values["frame_index"], np.arange(n))
                    or values["timestamp"].shape != (n,)
                    or not np.allclose(values["timestamp"], np.arange(n)/fps, atol=1e-4, rtol=0)):
                raise ValueError(f"Interaction sidecar temporal/state contract failed: {path}")
            self.cache[key] = values
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        values = self.cache[key]
        self.cache.move_to_end(key)
        if length is not None and len(values["state"]) != length:
            raise ValueError("Interaction sidecar episode length mismatch")
        return values

    def frame(self, task, episode, frame, fps, *, length, state, timestamp):
        values = self.episode(task, episode, fps, length)
        if not 0 <= frame < length:
            raise IndexError("Interaction sidecar frame outside episode")
        if (not np.allclose(values["state"][frame], state, atol=1e-5, rtol=0)
                or not np.isclose(values["timestamp"][frame], timestamp, atol=1e-5, rtol=0)):
            raise ValueError("Interaction sidecar does not match the current dataset state/timestamp")
        return values["interaction_geometry"][frame].copy(), values["interaction_valid"][frame].copy()


class InteractionRecorder:
    """Capture current geometry in the simulator thread, beside each RGB/state row.

    Call BEFORE applying the row's action. No simulator steps may occur between
    RGB/state capture and append(). Pass the raw 46D state and 44D rotvec action,
    in the SAME layout used by the LeRobot writer (before normalization).
    """
    def __init__(self, task):
        if task not in ENTITY_NAMES:
            raise ValueError(f"Unsupported interaction task: {task}")
        self.task = task
        self.rows = []

    def append(self, model, data, *, state, action, frame_index, timestamp):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if state.shape != (46,) or action.shape != (44,) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("Recorder requires finite RAW state[46] and rotvec action[44]")
        if frame_index != len(self.rows) or not np.isfinite(timestamp):
            raise ValueError("Recorder requires contiguous indices and finite timestamps")
        value, valid = extract_interaction_geometry(model, data, self.task)
        self.rows.append((value, valid, state.copy(), action.copy(), float(timestamp)))

    def save(self, path):
        if not self.rows:
            raise ValueError("Cannot save an empty interaction recording")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never silently replace another demonstration.
        with path.open("xb") as stream:
            np.savez_compressed(
                stream, schema=SCHEMA, task=self.task, label_source="synchronous_simulator_capture",
                interaction_geometry=np.stack([r[0] for r in self.rows]),
                interaction_valid=np.stack([r[1] for r in self.rows]),
                state=np.stack([r[2] for r in self.rows]),
                action=np.stack([r[3] for r in self.rows]),
                timestamp=np.asarray([r[4] for r in self.rows]),
                frame_index=np.arange(len(self.rows)),
            )
