"""Recover object pose recordings from synchronized full MuJoCo states, without rendering.

Each input episode-NNNNNN.npz needs qpos[T,nq], state[T,46], timestamp[T],
frame_index[T], and a matching episode-NNNNNN.xml with that episode's model
parameters (including reset-time body/table modifications). This does NOT
replay action commands or infer object motion from robot state.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from starVLA.object_pose import PhotographObjectPoseRecorder


def robot_state(data):
    parts = [data.sensor(f"franka/flange_{kind}_{side}").data.copy()
             for side in ("right", "left") for kind in ("pos", "quat")]
    for side, fingers in (("right", ("ff", "mf", "rf", "th")),
                          ("left", ("rf", "mf", "ff", "th"))):
        for finger in fingers:
            parts.extend(data.sensor(f"allegro_{side}/{finger}j{i}_pos").data.copy() for i in range(4))
    return np.concatenate(parts).astype(np.float32)


def main(raw_root: Path, output_root: Path):
    paths = sorted(raw_root.glob("episode-*.npz"))
    if not paths:
        raise FileNotFoundError(f"No episode state recordings under {raw_root}")
    for path in paths:
        model = mujoco.MjModel.from_xml_path(str(path.with_suffix(".xml")))
        data = mujoco.MjData(model)
        recorder = PhotographObjectPoseRecorder()
        with np.load(path, allow_pickle=False) as archive:
            qpos = archive["qpos"]
            count = len(qpos)
            if qpos.shape != (count, model.nq) or not np.isfinite(qpos).all():
                raise ValueError(f"Invalid full qpos shape/values: {path}")
            if archive["state"].shape != (count, 46) or archive["timestamp"].shape != (count,):
                raise ValueError(f"Invalid state/timestamp lengths: {path}")
            if not np.array_equal(archive["frame_index"], np.arange(count)):
                raise ValueError(f"Invalid frame indices: {path}")
            for index in range(count):
                data.qpos[:] = qpos[index]
                mujoco.mj_forward(model, data)
                state = robot_state(data)
                if not np.allclose(state, archive["state"][index], atol=1e-5, rtol=0):
                    raise ValueError(f"qpos/model does not reconstruct recorded robot state: {path}, frame {index}")
                recorder.append(model, data, state, index, archive["timestamp"][index])
        recorder.save(output_root / path.name)
        print(f"Recorded {path.stem}: {count} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    main(**vars(parser.parse_args()))
