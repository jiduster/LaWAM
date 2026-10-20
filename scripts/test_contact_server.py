"""CPU test of the DexJoCo server adapter; no checkpoint or websocket needed."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.serve_lawam_contact import ContactLaWAMPolicy, server


def main():
    policy = ContactLaWAMPolicy.__new__(ContactLaWAMPolicy)
    spec = SimpleNamespace(embodiment_id=28, stats_key="dexjoco_bimanual", robot_type="dual_arm")
    policy._resolve_robot_spec = lambda obs: spec
    policy._extract_primary_image = lambda obs: obs["base"]
    policy._extract_wrist_images = lambda obs, spec: [obs["wrist_left"], obs["wrist_right"]]
    policy._normalize_state = lambda state, spec: state * 0.5
    policy._maybe_log_action_summary = lambda *args: None
    policy.action_hz = 30
    policy.guidance_scale = 1.0
    policy.num_inference_steps = 2
    policy.stats_bounds = "min_max"
    policy.norm_stats = {"dexjoco_bimanual": {"action": {}}}
    captured = []

    def predict(examples, **kwargs):
        captured.extend(examples)
        return {"normalized_actions": np.zeros((30, 44), np.float32)}

    policy.model = SimpleNamespace(predict_action=predict)
    geometry = np.zeros((9, 6), np.float32)
    geometry[0, 5] = 1
    obs = {"state": np.ones(46), "prompt": "take a photograph", "contact_geometry": geometry}
    obs.update({key: np.zeros((8,8,3), np.uint8) for key in ("base", "wrist_left", "wrist_right")})
    with patch.object(server, "_unnormalize_min_max", side_effect=lambda actions, *a, **kw: actions + 1):
        result = policy.infer(obs)
    np.testing.assert_array_equal(captured[0]["contact_geometry"], geometry)
    np.testing.assert_array_equal(captured[0]["state"], np.ones(46) * 0.5)
    assert len(captured[0]["wrist_image"]) == 2
    assert result["actions"].shape == (30,44)
    assert (result["actions"] == 1).all()
    del obs["contact_geometry"]
    try:
        policy.infer(obs)
    except KeyError:
        pass
    else:
        raise AssertionError("Missing oracle input must fail")
    with patch.object(server.LaWAMPolicy, "__init__", return_value=None):
        policy.model.policy_backend = SimpleNamespace(flow=SimpleNamespace(contact_encoder=None))
        try:
            policy.__init__()
        except ValueError:
            pass
        else:
            raise AssertionError("Baseline checkpoint must be rejected by the oracle server")
    print("CONTACT_SERVER_TEST_OK")


if __name__ == "__main__":
    main()
