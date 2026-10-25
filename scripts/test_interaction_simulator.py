"""Real MuJoCo extraction checks; run in the DexJoCo simulator environment."""
import importlib
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from scripts.export_interaction_replay import PhysicsOnlyViewer, CLASSES
from starVLA.interaction_geometry import extract_interaction_geometry, InteractionRecorder
from scripts.prepare_interaction_candidates import convert, OBJECTS


class SimulatorGeometryTests(unittest.TestCase):
    def test_live_pose_and_recorder_for_both_tasks(self):
        for task,cls in CLASSES.items():
            module=importlib.import_module('dexjoco.sim.envs.panda_'+task+'_env')
            original=module.MujocoRenderer
            module.MujocoRenderer=PhysicsOnlyViewer
            env=None
            try:
                env=getattr(module,cls)(image_obs=False,randomize=False,randomize_dynamics=False)
                env._prime_rgb_array_renderer=lambda:None
                env.reset()
                before,valid=extract_interaction_geometry(env._model,env._data,task)
                if 'microwave' in task:
                    env._data.jnt('microjoint').qpos[:]=.5
                    mujoco.mj_forward(env._model,env._data)
                    value,mask=extract_interaction_geometry(env._model,env._data,task)
                    self.assertAlmostEqual(float(value[2,15]),.5)
                    self.assertAlmostEqual(float(value[3,15]),.5)
                    self.assertGreater(abs(value[3,:3]-before[3,:3]).max(),.01)
                else:
                    env._data.jnt('hanoi_disk_small_joint').qpos[0]+=.1
                    mujoco.mj_forward(env._model,env._data)
                    value,mask=extract_interaction_geometry(env._model,env._data,task)
                    self.assertAlmostEqual(float(value[3,0]-before[3,0]),.1,places=6)
                # Offline conversion must match the online Oracle schema, including
                # moving objects, door interaction points, handedness and extents.
                obs = env._compute_observation()['state']
                raw_state = np.r_[np.asarray(obs['tcp_pose']).ravel(), np.asarray(obs['gripper_pose']).ravel()]
                world = np.concatenate([np.r_[env._data.body(name).xpos, env._data.body(name).xquat]
                                        for name in OBJECTS[task]])[None]
                joint = np.array([[float(env._data.jnt('microjoint').qpos[0])]]) if 'microwave' in task else None
                offline, offline_mask = convert(task, world, raw_state[None], env._model, joint)
                np.testing.assert_allclose(offline[0], value, atol=2e-6, rtol=1e-6)
                np.testing.assert_array_equal(offline_mask[0], mask)
                recorder=InteractionRecorder(task)
                recorder.append(env._model,env._data,state=np.zeros(46),action=np.zeros(44),frame_index=0,timestamp=0)
                with tempfile.TemporaryDirectory() as tmp:
                    file=Path(tmp)/'episode-000000.npz'
                    recorder.save(file)
                    with np.load(file,allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['interaction_geometry'][0],value)
                        np.testing.assert_array_equal(z['interaction_valid'][0],mask)
                    with self.assertRaises(FileExistsError):
                        recorder.save(file)
            finally:
                if env is not None: env.close()
                module.MujocoRenderer=original


if __name__=='__main__':
    unittest.main()
