"""Official DexJoCo evaluator plus geometry captured synchronously with each RGB observation.

CLI matches dexjoco_openpi_client.eval_dexjoco_openpi. Safe to use for all four
variants: the three visual-only serving paths ignore the geometry fields.
"""
import tyro
from dexjoco_openpi_client import eval_dexjoco_openpi as evaluator
from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv
from starVLA.interaction_geometry import SCHEMA, ENTITY_NAMES, extract_interaction_geometry


class InteractionDexJoCoEnv(DexJoCoOpenPIEnv):
    def _process_obs(self,obs):
        if self.env_name not in ENTITY_NAMES:
            raise ValueError('Interaction evaluator supports Hanoi and microwave cook')
        result=super()._process_obs(obs)
        raw=self.env.unwrapped
        value,valid=extract_interaction_geometry(raw._model,raw._data,self.env_name)
        result.update(interaction_geometry=value,interaction_valid=valid,
                      interaction_task=self.env_name,interaction_schema=SCHEMA)
        return result


if __name__=='__main__':
    evaluator.DexJoCoOpenPIEnv=InteractionDexJoCoEnv
    tyro.cli(evaluator.main)
