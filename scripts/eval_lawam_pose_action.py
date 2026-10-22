"""DexJoCo evaluator adapter: copy pose with RGB/state in the simulator thread.

Original CLI, chunk execution, success decisions and video logging are preserved.
Use for BOTH matched baseline and orange runs with the same seeds.
"""
import tyro
from dexjoco_openpi_client import eval_dexjoco_openpi as evaluator
from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv
from starVLA.object_pose import extract_photograph_object_pose


class PoseActionDexJoCoEnv(DexJoCoOpenPIEnv):
    def start(self):
        if self.env_name != 'bimanual_photograph' or self.rand_full or self.randomize_dynamics:
            raise ValueError('Pose v1 requires bimanual_photograph with randomize=False, fixed dynamics')
        return super().start()

    def _process_obs(self, obs):
        if self.env_name != 'bimanual_photograph':
            raise ValueError('Object pose v1 only supports bimanual_photograph')
        result = super()._process_obs(obs)
        raw = self.env.unwrapped
        result['object_pose_condition'] = extract_photograph_object_pose(raw._model, raw._data)
        return result


if __name__ == '__main__':
    evaluator.DexJoCoOpenPIEnv = PoseActionDexJoCoEnv
    tyro.cli(evaluator.main)
