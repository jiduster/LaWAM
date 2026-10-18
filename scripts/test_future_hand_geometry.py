from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from scripts.test_external_latent_action_head import (
    TinyLAM, TinyProcessor, TinyVLM, build_tiny_backend, collate, raw_sample, wrap_backend,
)
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
from starVLA.model.framework.latent_world.runtime.output_mapper import map_policy_train_output
from starVLA.model.framework.latent_world.runtime.offline_diagnostics import diagnose_train_batch, masked_error_per_sample
from starVLA.model.framework.vlas.lawam import LatentWorldPolicyBackend
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups


CONFIG_PATH = Path(__file__).resolve().parents[1] / "starVLA/config/training/starvla_train_dexjoco_bimanual_photograph_C_h30_geometry.yaml"


def geometry_backend(*, inject=True, alternate=True, mode="vlm_queries"):
    cfg = copy.deepcopy(build_tiny_backend(mode).model_cfg)
    cfg.geometry_conditioning = True
    cfg.geometry_inject = inject
    cfg.geometry_frame_indices = (1, 2, 3)
    cfg.geometry_loss_weight = 0.05
    cfg.geometry_hidden_dim = 32
    cfg.flow_cfg.geometry_attention_every_n_blocks = 2
    cfg.flow_cfg.use_alternate_vldit = alternate
    with patch("starVLA.model.framework.vlas.lawam._load_vlm_and_processor", return_value=(TinyVLM(), TinyProcessor(), None, 99)), patch(
        "starVLA.model.framework.vlas.lawam.load_latent_action_model", return_value=TinyLAM(),
    ):
        return LatentWorldPolicyBackend(cfg, vlm_model_id="unused")


def geometry_sample():
    sample = raw_sample()
    sample["fingertip_positions"] = torch.arange(4 * 8 * 3).float().view(4, 8, 3) * 0.001
    sample["fingertip_valid"] = torch.ones(4, 8, dtype=torch.bool)
    return sample


def infer_batch(batch):
    result = {k: v for k, v in batch.items() if k not in ("primary_video", "actions", "actions_mask", "geometry_targets", "geometry_valid")}
    result["primary_image"] = batch["primary_video"][:, 0]
    return result


class FutureHandGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_collator_selects_exact_frames_masks_and_requires_sidecars(self):
        backend = geometry_backend()
        sample = geometry_sample()
        sample["fingertip_valid"][-1] = False
        batch = collate(backend, [sample])
        torch.testing.assert_close(batch["geometry_targets"][0], sample["fingertip_positions"][[1, 2, 3]])
        self.assertFalse(batch["geometry_valid"][0, -1].any())
        self.assertNotIn("fingertip_targets", batch)
        with self.assertRaisesRegex(KeyError, "sidecar"):
            collate(backend, [raw_sample()])

    def test_masked_nonfinite_labels_and_all_invalid_batch(self):
        head = geometry_backend().geometry_head
        prediction = torch.zeros(2, 3, 8, 3, requires_grad=True)
        targets = head.positions_in_meters(prediction.detach())
        targets[0, 0, 0] = float("nan")
        valid = torch.ones(2, 3, 8, dtype=torch.bool)
        loss, rmse, fraction = head.supervised_loss(prediction, targets, valid)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(rmse.item(), 0)
        self.assertLess(fraction.item(), 1)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        loss, rmse, fraction = head.supervised_loss(prediction, targets, valid & False)
        self.assertEqual((loss.item(), rmse.item(), fraction.item()), (0, 0, 0))

    def test_zero_init_keeps_pretrained_policy_and_all_parameters_have_gradients(self):
        baseline = build_tiny_backend("vlm_queries").eval()
        backend = geometry_backend().eval()
        loaded = backend.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(loaded.missing_keys)
        self.assertTrue(all(k.startswith(("geometry_head.", "flow.DiT.geometry_blocks.")) for k in loaded.missing_keys))
        batch = collate(backend, [geometry_sample(), geometry_sample()])
        torch.manual_seed(11)
        expected = baseline.predict_action(batch=infer_batch(batch))
        torch.manual_seed(11)
        actual = backend.predict_action(batch=infer_batch(batch))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        backend.train()
        losses = backend(batch=batch)
        losses["loss_total"].backward()
        for name, parameter in backend.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        mapped = map_policy_train_output(losses)
        self.assertIn("geometry_rmse_mm", mapped)
        self.assertGreater(backend.geometry_head.position_head[-1].weight.grad.abs().sum().item(), 0)
        output_projection = next(iter(backend.flow.DiT.geometry_blocks.values())).attention.to_out[0]
        self.assertGreater(output_projection.weight.grad.abs().sum().item(), 0)

    def test_bc_reaches_geometry_and_world_model_and_geometry_changes_actions(self):
        for alternate in (True, False):
            with self.subTest(alternate=alternate):
                backend = geometry_backend(alternate=alternate)
                for block in backend.flow.DiT.geometry_blocks.values():
                    torch.nn.init.normal_(block.attention.to_out[0].weight, std=0.02)
                batch = collate(backend, [geometry_sample(), geometry_sample()])
                losses = backend(batch=batch)
                losses["loss_flow"].backward()
                for module in (backend.geometry_head.position_head, backend.geometry_head.coordinate_encoder, backend.lam.decoder, backend.vlm_to_lam):
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None), 0)
                backend.eval()
                torch.manual_seed(15)
                original = backend.predict_action(batch=infer_batch(batch))
                backend.geometry_head.position_head[-1].bias.data.add_(1)
                torch.manual_seed(15)
                changed = backend.predict_action(batch=infer_batch(batch))
                self.assertGreater((original - changed).abs().max().item(), 1e-7)

    def test_future_images_and_labels_never_enter_predicted_geometry(self):
        backend = geometry_backend(mode="external_qformer").eval()
        batch = collate(backend, [geometry_sample()])
        predictions = []
        hook = backend.geometry_head.register_forward_hook(lambda module, args, output: predictions.append(output[0].detach()))
        backend(batch=batch)
        altered = dict(batch)
        altered["primary_video"] = batch["primary_video"].clone()
        altered["primary_video"][:, 1] += 10
        altered["geometry_targets"] = batch["geometry_targets"] + 10
        backend(batch=altered)
        torch.testing.assert_close(predictions[0], predictions[1])
        backend.predict_action(batch=infer_batch(batch), return_intermediates=True)
        torch.testing.assert_close(predictions[0], predictions[2])
        hook.remove()

    def test_auxiliary_only_has_no_unused_token_encoders(self):
        backend = geometry_backend(inject=False)
        self.assertFalse(backend.flow.DiT.geometry_blocks)
        self.assertFalse(hasattr(backend.geometry_head, "coordinate_encoder"))
        losses = backend(batch=collate(backend, [geometry_sample()]))
        losses["loss_total"].backward()
        for name, p in backend.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)

    def test_cfg_and_teacher_diagnostic_use_geometry(self):
        backend = geometry_backend()
        batch = collate(backend, [geometry_sample()])
        backend.flow.config.cfg_drop_prob = 1.0
        self.assertTrue(torch.isfinite(backend(batch=batch)["loss_total"]))
        backend.eval()
        for block in backend.flow.DiT.geometry_blocks.values():
            torch.nn.init.normal_(block.attention.to_out[0].weight, std=0.02)
        output = backend.predict_action(batch=infer_batch(batch), guidance_scale=1.5)
        self.assertTrue(torch.isfinite(output).all())
        metrics = diagnose_train_batch(backend, batch, noise_repeats=1, noise_seed=21)
        torch.manual_seed(21)
        normal = backend.predict_action(batch=infer_batch(batch), guidance_scale=1.0, return_padded=True)
        expected = masked_error_per_sample(normal, batch["actions"], batch["actions_mask"])
        torch.testing.assert_close(metrics["action_mse/predicted/all"], expected, atol=1e-6, rtol=1e-6)

    def test_config_validation_lr_groups_and_checkpoint_roundtrip(self):
        cfg = OmegaConf.load(CONFIG_PATH)
        policy = LatentWorldPolicyConfigBuilder(cfg).build()
        self.assertEqual(tuple(policy.geometry_frame_indices), (6, 15, 29))
        invalid = copy.deepcopy(cfg)
        invalid.framework.action_model.geometry_frame_indices = [6, 30]
        with self.assertRaisesRegex(ValueError, "offsets"):
            LatentWorldPolicyConfigBuilder(invalid).build()
        source = wrap_backend(geometry_backend())
        source.config = cfg
        groups = build_param_lr_groups(source, cfg)
        head_ids = {id(p) for p in source.policy_backend.geometry_head.parameters()}
        group = next(g for g in groups if head_ids.intersection(map(id, g["params"])))
        self.assertEqual(group["lr"], 1e-4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "final_model").mkdir()
            path = root / "final_model/pytorch_model.pt"
            torch.save(source.state_dict(), path)
            OmegaConf.save(cfg, root / "config.yaml")
            (root / "dataset_statistics.json").write_text(json.dumps({"dexjoco_bimanual": {}}))
            target = wrap_backend(geometry_backend())
            with patch("starVLA.model.framework.build_framework", return_value=target):
                restored = baseframework.from_pretrained(str(path))
            for name, value in source.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name])
            # A normal pretrained policy can initialize every original parameter,
            # leaving only the new geometry modules at their chosen initialization.
            pretrained = wrap_backend(build_tiny_backend("vlm_queries"))
            torch.save(pretrained.state_dict(), path)
            TrainerUtils.load_finetune_init_weights(target, str(path), load_pretrained_policy_lam=False)


if __name__ == "__main__":
    unittest.main()
