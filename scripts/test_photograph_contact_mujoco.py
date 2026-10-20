"""CPU-only checks against DexJoCo's actual photograph XML; no renderer/GPU."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from starVLA.contact_geometry import extract_photograph_contact
from scripts.record_photograph_contact_from_qpos import robot_state


def main(xml: Path):
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial = extract_photograph_contact(model, data)
    assert robot_state(data).shape == (46,)
    shutter = model.geom("camera_body_top_cap").id
    body_geom = model.geom("camera_body_collision").id
    # Normal must point away from the camera's collision body centre.
    outward = data.geom_xpos[shutter] - data.geom_xpos[body_geom]
    assert np.dot(initial[0, 3:], outward) > 0
    np.testing.assert_allclose(np.linalg.norm(initial[0, :3] - data.geom_xpos[shutter]),
                               model.geom_size[shutter, 1], atol=1e-7)
    address = model.jnt_qposadr[model.joint("camera_root").id]
    shift = np.array([0.1, -0.03, 0.02])
    data.qpos[address:address+3] += shift
    mujoco.mj_forward(model, data)
    moved = extract_photograph_contact(model, data)
    np.testing.assert_allclose(moved[0, :3] - initial[0, :3], shift, atol=1e-7)
    np.testing.assert_allclose(moved[0, 3:], initial[0, 3:], atol=1e-7)
    np.testing.assert_array_equal(moved[1:], initial[1:])

    old_position = data.qpos[address:address+3].copy()
    quat = data.qpos[address+3:address+7].copy()
    rotation = np.array([np.cos(0.3), 0, 0, np.sin(0.3)])
    composed = np.empty(4)
    mujoco.mju_mulQuat(composed, rotation, quat)
    data.qpos[address+3:address+7] = composed
    mujoco.mj_forward(model, data)
    rotated = extract_photograph_contact(model, data)
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, rotation)
    matrix = matrix.reshape(3, 3)
    np.testing.assert_allclose(rotated[0, :3], old_position + matrix @ (moved[0, :3]-old_position), atol=1e-7)
    np.testing.assert_allclose(rotated[0, 3:], matrix @ moved[0, 3:], atol=1e-7)

    # Exercise the actual evaluator adapter without starting a renderer.
    from scripts.eval_lawam_contact import ContactDexJoCoEnv
    wrapper = ContactDexJoCoEnv.__new__(ContactDexJoCoEnv)
    wrapper.env_name = "bimanual_photograph"
    wrapper.env = SimpleNamespace(unwrapped=SimpleNamespace(_model=model, _data=data))
    wrapper.camera_mapping = {"base": "ego"}
    wrapper.dual_arm = True
    wrapper.prompt = "take a photograph"
    result = wrapper._process_obs({"ego": np.zeros((32,32,3), np.uint8), "state": robot_state(data)})
    np.testing.assert_array_equal(result["contact_geometry"], rotated)
    print("PHOTOGRAPH_CONTACT_MUJOCO_OK: moving target, rigid rotation, outward surface, live observation adapter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=Path(
        "/data/home/zyh/dexjoco/dexjoco/dexjoco/sim/envs/xmls/arena_arm_hand_bimanual_photograph.xml"))
    main(**vars(parser.parse_args()))
