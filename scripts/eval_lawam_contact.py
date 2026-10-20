"""DexJoCo evaluator with synchronized current oracle contact observations.

CLI matches dexjoco_openpi_client.eval_dexjoco_openpi. Environment lives in the
main process; the existing inference worker receives geometry with the image.
"""

import tyro
from dexjoco_openpi_client import eval_dexjoco_openpi as evaluator
from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv
from starVLA.contact_geometry import extract_photograph_contact


class ContactDexJoCoEnv(DexJoCoOpenPIEnv):
    def _process_obs(self, obs):
        if self.env_name != "bimanual_photograph":
            raise ValueError("Oracle shutter schema only supports bimanual_photograph")
        result = super()._process_obs(obs)
        raw_env = self.env.unwrapped
        result["contact_geometry"] = extract_photograph_contact(raw_env._model, raw_env._data)
        return result


if __name__ == "__main__":
    evaluator.DexJoCoOpenPIEnv = ContactDexJoCoEnv
    tyro.cli(evaluator.main)
