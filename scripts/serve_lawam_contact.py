"""OpenPI server adapter that passes current oracle geometry into LaWAM.

Use the same CLI arguments as dexjoco_openpi_client.cli.serve_lawam.
"""

import numpy as np
import tyro

from dexjoco_openpi_client import lawam_policy_server as server
from starVLA.contact_geometry import validate_geometry


class ContactLaWAMPolicy(server.LaWAMPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        flow = self.model.policy_backend.flow
        if flow.contact_encoder is None or not flow.DiT.contact_attention:
            raise ValueError("This server requires a contact-enabled checkpoint; use the original server for control")
        server.LOGGER.info("Oracle contact enabled: 17 tokens, residual attention after blocks %s",
                           [int(index) + 1 for index in flow.DiT.contact_attention])

    def infer(self, obs):
        # The baseline server constructs its own example and drops extra keys.
        # Keep normalization/image/action conventions identical to that server.
        spec = self._resolve_robot_spec(obs)
        raw_state = np.asarray(obs.get("state", []), dtype=np.float32).reshape(-1)
        if "contact_geometry" not in obs:
            raise KeyError("Use scripts/eval_lawam_contact.py: oracle policy requires live contact_geometry")
        geometry = validate_geometry(obs["contact_geometry"])
        if geometry.shape != (9, 6):
            raise ValueError("Expected one current contact frame [9,6]")
        example = {
            "primary_image": [self._extract_primary_image(obs)],
            "lang": str(obs.get("prompt", "")),
            "embodiment_id": spec.embodiment_id,
            "action_hz": self.action_hz,
            "state": self._normalize_state(raw_state, spec),
            "contact_geometry": geometry,
        }
        wrists = self._extract_wrist_images(obs, spec)
        if wrists:
            example["wrist_image"] = wrists
        result = self.model.predict_action([example], guidance_scale=self.guidance_scale,
                                           num_inference_steps=self.num_inference_steps)
        actions = server._unnormalize_min_max(result["normalized_actions"],
                                              self.norm_stats[spec.stats_key]["action"],
                                              bounds=self.stats_bounds)
        self._maybe_log_action_summary(actions, raw_state, spec)
        return {"actions": actions, "lawam_metadata": {
            "robot_type": spec.robot_type, "stats_key": spec.stats_key,
            "embodiment_id": spec.embodiment_id, "oracle_contact": True,
        }}


if __name__ == "__main__":
    server.LaWAMPolicy = ContactLaWAMPolicy
    tyro.cli(server.serve_lawam)
