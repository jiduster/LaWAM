import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.test_external_latent_action_head import build_tiny_backend, raw_sample, collate, wrap_backend
from scripts.prepare_photograph_contact import prepare
from starVLA.contact_geometry import SCHEMA, FINGER_ORDER, ContactStore
from starVLA.model.framework.base_framework import validate_contact_checkpoint_keys
from starVLA.model.framework.vlas.contact_conditioning import ContactEncoder
from starVLA.model.framework.latent_world.batch_builder import LatentWorldPolicyInferBatchBuilder
from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
from starVLA.model.framework.latent_world.vlm_adapter import LatentWorldPolicyVLMAdapter
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups


CONFIG = "starVLA/config/training/starvla_train_dexjoco_bimanual_photograph_C_h30_contact.yaml"


def geometry(batch=1):
    g = torch.zeros(batch, 9, 6)
    g[:, :, :3] = torch.randn(batch, 9, 3) * 0.1
    g[:, 0, 5] = 1
    return g


def make_batch(backend, count=1):
    # No store is needed for these synthetic model tests.
    backend.model_cfg.flow_cfg.contact_conditioning = False
    batch = collate(backend, [raw_sample() for _ in range(count)])
    backend.model_cfg.flow_cfg.contact_conditioning = True
    batch["contact_geometry"] = geometry(count)
    return batch


def infer_batch(batch):
    result = {k: v for k, v in batch.items() if k not in ("primary_video", "actions", "actions_mask")}
    result["primary_image"] = batch["primary_video"][:, 0]
    return result


class OracleContactTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        torch.set_num_threads(1)

    def test_zero_init_preserves_baseline_both_dit_variants_and_cfg(self):
        for alternate in (True, False):
            baseline = build_tiny_backend(mode="vlm_queries", alternate=alternate).eval()
            oracle = build_tiny_backend(mode="vlm_queries", alternate=alternate, contact=True).eval()
            result = oracle.load_state_dict(baseline.state_dict(), strict=False)
            self.assertFalse(result.unexpected_keys)
            self.assertTrue(all("contact_" in key for key in result.missing_keys))
            batch = make_batch(oracle)
            for scale in (1.0, 1.5):
                torch.manual_seed(9)
                expected = baseline.predict_action(batch=infer_batch(batch), guidance_scale=scale)
                torch.manual_seed(9)
                actual = oracle.predict_action(batch=infer_batch(batch), guidance_scale=scale)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_bc_gradients_and_action_sensitivity_after_update(self):
        for alternate in (True, False):
            model = build_tiny_backend(mode="vlm_queries", alternate=alternate, contact=True)
            model.model_cfg.repeated_diffusion_steps = 2
            batch = make_batch(model, count=2)
            loss = model(batch=batch)["loss_flow"]
            loss.backward()
            params = [(n, p) for n, p in model.flow.named_parameters() if "contact_" in n]
            self.assertTrue(all(p.grad is not None for _, p in params))
            self.assertGreater(sum(p.grad.abs().sum().item() for n, p in params if "out_proj" in n), 0)
            opt = torch.optim.Adam([p for _, p in params], lr=0.01)
            opt.step()
            opt.zero_grad(set_to_none=True)
            model(batch=batch)["loss_flow"].backward()
            self.assertGreater(model.flow.contact_encoder.target[0].weight.grad.abs().sum().item(), 0)
            model.eval()
            original = infer_batch(batch)
            altered = copy.deepcopy(original)
            altered["contact_geometry"][:, 0, 0] += 0.1
            torch.manual_seed(9)
            a = model.predict_action(batch=original)
            torch.manual_seed(9)
            b = model.predict_action(batch=altered)
            self.assertGreater((a-b).abs().max().item(), 1e-7)
            restored = build_tiny_backend(mode="vlm_queries", alternate=alternate, contact=True).eval()
            restored.load_state_dict(model.state_dict(), strict=True)
            torch.manual_seed(9)
            torch.testing.assert_close(a, restored.predict_action(batch=original))
            del original["contact_geometry"]
            with self.assertRaisesRegex(KeyError, "contact_geometry"):
                model.predict_action(batch=original)

    def test_invalid_geometry_rejected(self):
        enc = ContactEncoder(16)
        g = geometry()
        self.assertEqual(enc(g).shape, (1, 17, 16))
        for bad in (g[:, :8], g * float("nan"), g * 0):
            with self.assertRaises(ValueError):
                enc(bad)

    def test_inference_checkpoint_requires_contact_config_and_weights_to_agree(self):
        key = "policy_backend.flow.contact_encoder.identity"
        validate_contact_checkpoint_keys({key}, {key})
        validate_contact_checkpoint_keys(set(), set())
        with self.assertRaises(RuntimeError):
            validate_contact_checkpoint_keys({key}, set())
        with self.assertRaises(RuntimeError):
            validate_contact_checkpoint_keys(set(), {key})

    def test_current_sidecar_collation_and_inference_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bimanual_photograph"
            (root / "episodes").mkdir(parents=True)
            g = geometry(3).numpy()
            np.savez(root / "episodes/episode-000000.npz", contact_geometry=g, frame_index=np.arange(3))
            (root / "manifest.json").write_text(json.dumps(dict(
                schema=SCHEMA, dataset=root.name, finger_order=FINGER_ORDER,
                fps=30, episodes={"0": 3})))
            backend = build_tiny_backend(mode="vlm_queries", contact=True)
            backend.model_cfg.contact_sidecar_root = tmp
            sample = raw_sample()
            sample.update(dataset_name=root.name, episode_index=0, frame_index=1)
            batch = collate(backend, [sample])
            np.testing.assert_array_equal(batch["contact_geometry"][0].numpy(), g[1])
            # A future row must not enter the current observation.
            g[2, :, :3] += 100
            np.savez(root / "episodes/episode-000000.npz", contact_geometry=g, frame_index=np.arange(3))
            torch.testing.assert_close(batch["contact_geometry"], collate(backend, [sample])["contact_geometry"])
            adapter = LatentWorldPolicyVLMAdapter(model=backend.vlm, processor=backend.processor,
                config=OmegaConf.load(CONFIG), placeholder_token="<ACT_PH>", act_queries=8, flow_queries=8)
            builder = LatentWorldPolicyInferBatchBuilder(policy_cfg=backend.model_cfg,
                policy_backend=backend, policy_vlm_adapter=adapter)
            ex = dict(lang=sample["lang"], primary_image=[np.zeros((8,8,3), np.uint8)],
                      action_hz=30, embodiment_id=28, contact_geometry=g[1])
            torch.testing.assert_close(builder.build_infer_batch([ex])["contact_geometry"], batch["contact_geometry"])
            del ex["contact_geometry"]
            with self.assertRaises(KeyError):
                builder.build_infer_batch([ex])

    def test_config_and_new_parameter_lr(self):
        cfg = OmegaConf.load(CONFIG)
        parsed = LatentWorldPolicyConfigBuilder(cfg).build()
        self.assertTrue(parsed.flow_cfg.contact_conditioning)
        model = wrap_backend(build_tiny_backend(mode="vlm_queries", contact=True))
        groups = build_param_lr_groups(model, cfg)
        lrs = {id(p): group["lr"] for group in groups for p in group["params"]}
        for name, param in model.policy_backend.flow.named_parameters():
            self.assertEqual(lrs[id(param)], 1e-4 if "contact_" in name else 1e-5)

    def test_recordings_must_match_dataset_and_hashes(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bimanual_photograph"
            (root / "data").mkdir(parents=True)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text('{"fps":30}')
            state = np.zeros((2,46), np.float32)
            table = pa.table(dict(episode_index=[0,0], frame_index=[0,1], timestamp=[0.,1/30],
                                 **{"observation.state": state.tolist()}))
            parquet = root / "data/test.parquet"
            pq.write_table(table, parquet)
            recordings = Path(tmp) / "raw"
            recordings.mkdir()
            path = recordings / "episode-000000.npz"
            args = dict(schema=SCHEMA, contact_geometry=geometry(2).numpy(), frame_index=np.arange(2),
                        state=state, timestamp=np.array([0.,1/30]))
            np.savez(path, **dict(args, state=state+1))
            with self.assertRaisesRegex(ValueError, "state mismatch"):
                prepare(root, Path(tmp)/"bad", recordings)
            np.savez(path, **args)
            out = Path(tmp)/"sidecars"
            prepare(root, out, recordings)
            prepare(root, out, validate_only=True)
            with self.assertRaises(ValueError):
                ContactStore(out).frame(root.name, 0, 0, 60)
            with self.assertRaises(IndexError):
                ContactStore(out).frame(root.name, 0, -1, 30)
            pq.write_table(table.slice(0,1), parquet)
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare(root, out, validate_only=True)


if __name__ == "__main__":
    unittest.main()
