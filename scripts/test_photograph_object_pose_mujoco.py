"""Validate the actual photograph XML and observation adapter without rendering."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from dexjoco.sim.envs.panda_bimanual_photograph_env import PandaBimanualPhotographGymEnv
from scripts.eval_lawam_pose_action import PoseActionDexJoCoEnv
from scripts.record_photograph_object_pose_from_qpos import robot_state
from starVLA.object_pose import extract_photograph_object_pose


def main(xml):
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    env = PandaBimanualPhotographGymEnv.__new__(PandaBimanualPhotographGymEnv)
    env._model, env._data = model,data
    env._target_region_geom_id = model.geom('photograph_target_region_visual').id
    env._view_site_id = model.site('view_direction_site').id
    env._camera_axis_z_geom_id = model.geom('camera_axis_z').id
    env._shutter_geom_id = model.geom('camera_body_top_cap').id
    env._camera_geom_ids = {env._shutter_geom_id}
    initial = extract_photograph_object_pose(model,data)
    camera = model.camera('back').id
    r_we = data.cam_xmat[camera].reshape(3,3) @ np.diag([1.,-1.,-1.])
    np.testing.assert_allclose(data.cam_xpos[camera] + r_we @ initial[:3],data.site_xpos[env._view_site_id],atol=2e-7)
    r_eo = r_we.T @ data.geom_xmat[env._camera_axis_z_geom_id].reshape(3,3)
    np.testing.assert_allclose(initial[3:9],np.concatenate((r_eo[:,0],r_eo[:,1])),atol=1e-7)
    joint = model.jnt_qposadr[model.joint('camera_root').id]
    original = data.qpos[joint:joint+7].copy()
    shift=np.array([.1,-.03,.02]); data.qpos[joint:joint+3]+=shift
    mujoco.mj_forward(model,data)
    moved=extract_photograph_object_pose(model,data)
    np.testing.assert_allclose(moved[:3]-initial[:3],r_we.T @ shift,atol=1e-7)
    np.testing.assert_array_equal(moved[3:],initial[3:])
    maximum_error=0
    for angle in np.linspace(-2.5,2.5,11):
        rotation=np.array([np.cos(angle/2),0,0,np.sin(angle/2)])
        composed=np.empty(4); mujoco.mju_mulQuat(composed,rotation,original[3:])
        data.qpos[joint+3:joint+7]=composed
        mujoco.mj_forward(model,data)
        value=extract_photograph_object_pose(model,data)
        _,metrics=env._compute_success_metrics()
        pose_angle=np.degrees(np.arccos(np.clip(float(value[11]),-1,1)))
        maximum_error=max(maximum_error,abs(pose_angle-metrics['view_angle_deg']))
        np.testing.assert_allclose(pose_angle,metrics['view_angle_deg'],atol=2e-5)
    wrapper=PoseActionDexJoCoEnv.__new__(PoseActionDexJoCoEnv)
    wrapper.env_name='bimanual_photograph'
    wrapper.env=SimpleNamespace(unwrapped=SimpleNamespace(_model=model,_data=data))
    wrapper.camera_mapping={'base':'ego'}; wrapper.dual_arm=True; wrapper.prompt='take a photograph'
    snapshot=wrapper._process_obs({'ego':np.zeros((32,32,3),np.uint8),'state':robot_state(data)})
    np.testing.assert_array_equal(snapshot['object_pose_condition'],value)
    copied=snapshot['object_pose_condition'].copy()
    data.qpos[joint]+=.2; mujoco.mj_forward(model,data)
    np.testing.assert_array_equal(snapshot['object_pose_condition'],copied)
    print(f'PHOTOGRAPH_POSE_MUJOCO_OK translations, rotations, optical axes, copied observation; max_angle_error_deg={maximum_error:.8g}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml',type=Path,default=Path('/data/home/zyh/dexjoco/dexjoco/dexjoco/sim/envs/xmls/arena_arm_hand_bimanual_photograph.xml'))
    main(**vars(parser.parse_args()))
