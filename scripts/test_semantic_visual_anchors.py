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
from scripts.test_future_hand_geometry import geometry_sample, infer_batch
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
from starVLA.model.framework.latent_world.runtime.offline_diagnostics import diagnose_train_batch, masked_error_per_sample
from starVLA.model.framework.vlas.lawam import LatentWorldPolicyBackend
from starVLA.model.framework.vlas.semantic_visual_anchors import SemanticVisualAnchors
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups


CONFIG = Path(__file__).resolve().parents[1] / (
    "starVLA/config/training/starvla_train_dexjoco_bimanual_photograph_C_h30_visual_anchors.yaml"
)


def anchor_backend(*, alternate=True, geometry=False, mode="vlm_queries"):
    cfg = copy.deepcopy(build_tiny_backend(mode).model_cfg)
    cfg.visual_anchor_conditioning = True
    cfg.visual_anchor_num_tokens = 3
    cfg.visual_anchor_hidden_dim = 16
    cfg.visual_anchor_kmeans_iters = 2
    cfg.flow_cfg.anchor_attention_every_n_blocks = 2
    cfg.flow_cfg.use_alternate_vldit = alternate
    if geometry:
        cfg.geometry_conditioning = True
        cfg.geometry_frame_indices = (1, 2, 3)
        cfg.geometry_loss_weight = 0.05
        cfg.geometry_hidden_dim = 16
        cfg.flow_cfg.geometry_attention_every_n_blocks = 2
    with patch("starVLA.model.framework.vlas.lawam._load_vlm_and_processor", return_value=(TinyVLM(), TinyProcessor(), None, 99)), patch(
        "starVLA.model.framework.vlas.lawam.load_latent_action_model", return_value=TinyLAM(),
    ):
        return LatentWorldPolicyBackend(cfg, vlm_model_id="unused")


class SemanticVisualAnchorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_proposals_cover_flat_features_and_are_deterministic(self):
        head = SemanticVisualAnchors(
            vision_dim=8, vlm_dim=32, state_dim=46, token_dim=8,
            grid_height=3, grid_width=4, num_tokens=6,
        )
        features = torch.ones(2, 12, 8)
        weights, uv, indices = head.propose(features)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 6))
        torch.testing.assert_close(uv, head.patch_uv[indices])
        self.assertTrue((uv.abs() < 1).all())
        self.assertEqual(len(indices[0].unique()), 6)
        torch.manual_seed(100)
        repeated = head.propose(features)
        for expected, actual in zip((weights, uv, indices), repeated):
            torch.testing.assert_close(expected, actual)
        with self.assertRaisesRegex(ValueError, "DINO tokens"):
            head.propose(features[:, :-1])

    def test_task_state_masks_and_conditioning(self):
        head = anchor_backend().visual_anchor_head
        arguments = dict(
            current_features=torch.randn(2, 4, 8), semantic_features=torch.randn(2, 6, 32),
            semantic_mask=torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0]]),
            state=torch.randn(2, 46), state_mask=torch.ones(2, 46, dtype=torch.bool),
        )
        arguments["state_mask"][:, 20:] = False
        original, _ = head(**arguments)
        altered = {key: value.clone() for key, value in arguments.items()}
        altered["semantic_features"][~altered["semantic_mask"].bool()] = float("nan")
        altered["state"][~altered["state_mask"]] = float("nan")
        torch.testing.assert_close(original, head(**altered)[0])
        altered["state"][:, :20] += 1
        self.assertGreater((original - head(**altered)[0]).abs().max().item(), 1e-6)
        altered = dict(arguments, semantic_features=arguments["semantic_features"] + 2)
        altered["semantic_features"][..., 0] += 3
        self.assertGreater((original - head(**altered)[0]).abs().max().item(), 1e-6)

    def test_zero_init_matches_baseline_and_bc_reaches_every_new_parameter(self):
        baseline = build_tiny_backend("vlm_queries").eval()
        backend = anchor_backend().eval()
        result = backend.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(result.missing_keys)
        self.assertTrue(all(key.startswith(("visual_anchor_head.", "flow.DiT.anchor_blocks.")) for key in result.missing_keys))
        self.assertFalse(result.unexpected_keys)
        batch = collate(backend, [raw_sample(), raw_sample()])
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
        block = next(iter(backend.flow.DiT.anchor_blocks.values()))
        self.assertGreater(block.attention.to_out[0].weight.grad.abs().sum().item(), 0)
        # After the first output-projection update, gradients reach the encoders.
        torch.nn.init.normal_(block.attention.to_out[0].weight, std=0.02)
        backend.zero_grad(set_to_none=True)
        backend(batch=batch)["loss_flow"].backward()
        for name, parameter in backend.visual_anchor_head.named_parameters():
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_actions_change_when_anchor_tokens_change_both_dit_variants(self):
        for alternate in (False, True):
            with self.subTest(alternate=alternate):
                backend = anchor_backend(alternate=alternate).eval()
                for block in backend.flow.DiT.anchor_blocks.values():
                    torch.nn.init.normal_(block.attention.to_out[0].weight, std=0.05)
                batch = infer_batch(collate(backend, [raw_sample()]))
                torch.manual_seed(12)
                reference = backend.predict_action(batch=batch)

                def replace_tokens(module, args, output):
                    tokens, info = output
                    return torch.zeros_like(tokens), info

                hook = backend.visual_anchor_head.register_forward_hook(replace_tokens)
                torch.manual_seed(12)
                changed = backend.predict_action(batch=batch)
                hook.remove()
                self.assertGreater((reference - changed).abs().max().item(), 1e-7)

    def test_current_only_training_inference_and_diagnostic_parity(self):
        backend = anchor_backend(mode="external_qformer").eval()
        batch = collate(backend, [raw_sample()])
        recorded = []
        hook = backend.visual_anchor_head.register_forward_hook(lambda module, args, output: recorded.append(output[0].detach().clone()))
        backend(batch=batch)
        altered = dict(batch)
        altered["primary_video"] = batch["primary_video"].clone()
        altered["primary_video"][:, 1] += 10
        altered["actions"] = batch["actions"] + 10
        backend(batch=altered)
        torch.testing.assert_close(recorded[0], recorded[1])
        _, info = backend.predict_action(batch=infer_batch(batch), return_intermediates=True)
        torch.testing.assert_close(recorded[0], recorded[2])
        self.assertEqual(info["visual_anchor_uv"].shape, (1, 3, 2))
        torch.testing.assert_close(info["visual_anchor_importance"].sum(dim=1), torch.ones(1))
        hook.remove()
        metrics = diagnose_train_batch(backend, batch, noise_repeats=1, noise_seed=21)
        torch.manual_seed(21)
        prediction = backend.predict_action(batch=infer_batch(batch), return_padded=True)
        expected = masked_error_per_sample(prediction, batch["actions"], batch["actions_mask"])
        torch.testing.assert_close(metrics["action_mse/predicted/all"], expected, atol=1e-6, rtol=1e-6)

    def test_cfg_and_geometry_can_be_enabled_together(self):
        backend = anchor_backend(geometry=True)
        batch = collate(backend, [geometry_sample(), geometry_sample()])
        backend.flow.config.cfg_drop_prob = 1.0
        backend(batch=batch)["loss_total"].backward()
        for name, parameter in backend.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
        backend.eval()
        self.assertTrue(torch.isfinite(backend.predict_action(batch=infer_batch(batch), guidance_scale=1.5)).all())

    def test_saved_config_loading_lr_groups_and_finetune_initialization(self):
        cfg = OmegaConf.load(CONFIG)
        policy = LatentWorldPolicyConfigBuilder(cfg).build()
        self.assertTrue(policy.visual_anchor_conditioning)
        bad_cfg = copy.deepcopy(cfg)
        bad_cfg.framework.action_model.visual_anchor_num_tokens = 0
        with self.assertRaisesRegex(ValueError, "positive integer"):
            LatentWorldPolicyConfigBuilder(bad_cfg).build()
        source = wrap_backend(anchor_backend())
        source.config = cfg
        groups = build_param_lr_groups(source, cfg)
        new_ids = {id(p) for module in (source.policy_backend.visual_anchor_head, source.policy_backend.flow.DiT.anchor_blocks)
                   for p in module.parameters()}
        custom = next(group for group in groups if group["name"] == "visual_anchors")
        self.assertEqual(new_ids, {id(p) for p in custom["params"]})
        self.assertEqual(custom["lr"], 1e-4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "final_model").mkdir()
            path = root / "final_model/pytorch_model.pt"
            # Use a nonzero trained-like attention to prove actual weights restore.
            for block in source.policy_backend.flow.DiT.anchor_blocks.values():
                torch.nn.init.normal_(block.attention.to_out[0].weight, std=0.02)
            torch.save(source.state_dict(), path)
            OmegaConf.save(cfg, root / "config.yaml")
            (root / "dataset_statistics.json").write_text(json.dumps({"dexjoco_bimanual": {}}))
            target = wrap_backend(anchor_backend())
            with patch("starVLA.model.framework.build_framework", return_value=target):
                restored = baseframework.from_pretrained(str(path))
            for name, value in source.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name])
            source.eval()
            restored.eval()
            batch = infer_batch(collate(source.policy_backend, [raw_sample()]))
            torch.manual_seed(22)
            expected = source.policy_backend.predict_action(batch=batch)
            torch.manual_seed(22)
            torch.testing.assert_close(expected, restored.policy_backend.predict_action(batch=batch))
            baseline = wrap_backend(build_tiny_backend("vlm_queries"))
            torch.save(baseline.state_dict(), path)
            TrainerUtils.load_finetune_init_weights(target, str(path), load_pretrained_policy_lam=True)
            for name, value in baseline.state_dict().items():
                torch.testing.assert_close(value, target.state_dict()[name])


if __name__ == "__main__":
    unittest.main()
