"""Same-frame photograph task pose in the fixed back-camera optical frame."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

SCHEMA = "photograph_task_pose_back_optical_v1"


def validate_object_pose(value):
    pose = np.asarray(value, dtype=np.float32)
    if pose.ndim < 1 or pose.shape[-1] != 12 or not np.isfinite(pose).all():
        raise ValueError("Expected finite object_pose_condition [...,12]")
    if not np.allclose(np.linalg.norm(pose[..., 9:12], axis=-1), 1, atol=1e-3, rtol=0):
        raise ValueError("Object goal direction must be unit length")
    return pose


def extract_photograph_object_pose(model, data):
    """Read after mj_forward/step, in the RGB/state observation thread; copy now."""
    camera = model.camera("back").id
    # A tracking camera would violate the fixed-frame data contract.
    if int(model.cam_mode[camera]) != 0:
        raise ValueError("Pose conditioning requires a fixed back camera")
    p_we = data.cam_xpos[camera].copy()
    r_we = data.cam_xmat[camera].reshape(3, 3) @ np.diag([1., -1., -1.])
    p_wo = data.site_xpos[model.site("view_direction_site").id].copy()
    r_wo = data.geom_xmat[model.geom("camera_axis_z").id].reshape(3, 3).copy()
    r_eo = r_we.T @ r_wo
    return validate_object_pose(np.concatenate((r_we.T @ (p_wo - p_we),
                                               r_eo[:, 0], r_eo[:, 1],
                                               r_wo.T @ np.array([1., 0., 0.])))).copy()


class ObjectPoseStore:
    def __init__(self, root):
        self.root = Path(root) if str(root).strip() else None
        self.cache = OrderedDict()
        self.manifests = {}

    def manifest(self, dataset):
        if self.root is None:
            raise ValueError("object_pose_sidecar_root is required when samples have no object_pose_condition")
        if Path(dataset).name != dataset:
            raise ValueError("dataset must be a directory basename")
        if dataset not in self.manifests:
            with (self.root / dataset / "manifest.json").open() as stream:
                meta = json.load(stream)
            if meta["schema"] != SCHEMA or meta["dataset"] != dataset:
                raise ValueError("Object pose sidecar schema/dataset mismatch")
            self.manifests[dataset] = meta
        return self.manifests[dataset]

    def frame(self, dataset, episode, frame, fps):
        meta = self.manifest(dataset)
        if not np.isclose(meta["fps"], fps):
            raise ValueError("Object pose sidecar FPS mismatch")
        key = (dataset, int(episode))
        if key not in self.cache:
            path = self.root / dataset / "episodes" / f"episode-{episode:06d}.npz"
            with np.load(path, allow_pickle=False) as archive:
                pose = validate_object_pose(archive["object_pose_condition"]).copy()
                if pose.shape != (meta["episodes"][str(episode)], 12):
                    raise ValueError(f"Object pose sidecar length mismatch: {path}")
                if not np.array_equal(archive["frame_index"], np.arange(len(pose))):
                    raise ValueError(f"Object pose frame index mismatch: {path}")
            self.cache[key] = pose
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        if not 0 <= frame < len(self.cache[key]):
            raise IndexError(f"Object pose frame out of range: {key}, {frame}")
        return self.cache[key][frame].copy()


class PhotographObjectPoseRecorder:
    """Call append alongside each raw demonstration image/state, before stepping."""
    def __init__(self):
        self.rows = []

    def append(self, model, data, state, frame_index, timestamp):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (46,) or not np.isfinite(state).all():
            raise ValueError("Pose recorder requires finite raw state[46]")
        if frame_index != len(self.rows) or not np.isfinite(timestamp):
            raise ValueError("Pose recorder requires contiguous frames and finite timestamps")
        self.rows.append((extract_photograph_object_pose(model, data), state.copy(), float(timestamp)))

    def save(self, path):
        if not self.rows:
            raise ValueError("Cannot save empty object pose recording")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, schema=SCHEMA,
                                object_pose_condition=np.stack([r[0] for r in self.rows]),
                                state=np.stack([r[1] for r in self.rows]),
                                timestamp=np.array([r[2] for r in self.rows]),
                                frame_index=np.arange(len(self.rows)))
