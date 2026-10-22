from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from starVLA.dataloader import _build_latent_world_collator
from starVLA.model.framework.base_framework import baseframework, validate_full_checkpoint_state_dict
from starVLA.model.framework.latent_world.batch_builder import LatentWorldPolicyInferBatchBuilder
from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
from starVLA.model.framework.latent_world.runtime.output_mapper import map_policy_train_output
from starVLA.model.framework.latent_world.vlm_adapter import LatentWorldPolicyVLMAdapter
from starVLA.model.framework.vlas.flowmatching_expert import ConditionalFlowMatchingConfig
from starVLA.model.framework.vlas.latent_action_head import LatentActionHead
from starVLA.model.framework.vlas.lawam import LatentWorldPolicyBackend, LatentWorldPolicyConfig
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups


CONFIG_PATH = Path(__file__).resolve().parents[1] / (
    "starVLA/config/training/starvla_train_dexjoco_bimanual_photograph_external_latent.yaml"
)


class TinyProcessor:
    def apply_chat_template(self, messages, **kwargs):
        sequences = []
        for message in messages:
            sequence = []
            image_count = 0
            for item in message[0]["content"]:
                if item["type"] == "image":
                    image_count += 1
                    sequence.append(image_count)
                else:
                    sequence.append(4)
                    sequence.extend([99] * item["text"].count("<ACT_PH>"))
            sequence.append(5)
            sequences.append(sequence)
        length = max(map(len, sequences))
        return {
            "input_ids": torch.tensor([s + [0] * (length - len(s)) for s in sequences]),
            "attention_mask": torch.tensor([[1] * len(s) + [0] * (length - len(s)) for s in sequences]),
            "pixel_values": torch.zeros(len(sequences), 3, 8, 8),
        }


class TinyMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(32, 32)

    def forward(self, *, inputs_embeds, attention_mask, **kwargs):
        valid = attention_mask.unsqueeze(-1)
        context = (inputs_embeds * valid).cumsum(dim=1) / valid.cumsum(dim=1).clamp_min(1)
        return SimpleNamespace(last_hidden_state=self.projection(context))


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(128, 32)
        self.model = TinyMultimodalModel()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=32))

    def get_input_embeddings(self):
        return self.embedding


class TinyFutureDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_projection = nn.Linear(32, 8)

    def forward(self, current_features, latent):
        return current_features + self.latent_projection(latent)


class TinyLAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.code_dim = 32
        self.input_dim = 8
        self.encoder = SimpleNamespace(num_frames=2, grid_height=2, grid_width=2)
        self.vision_projection = nn.Linear(3, 8)
        self.latent_encoder = nn.Linear(8, 32)
        self.decoder = TinyFutureDecoder()
        self.requires_grad_(False)
        self.decoder.requires_grad_(True)

    def extract_vision_features(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(1)
        features = self.vision_projection(images.mean(dim=(-2, -1)))
        return features.unsqueeze(2).expand(-1, -1, 4, -1)

    def get_latent_action(self, *, videos, **kwargs):
        features = self.extract_vision_features(videos)
        delta = (features[:, -1] - features[:, 0]).mean(dim=1)
        return {"quantized": self.latent_encoder(delta).unsqueeze(1)}


def build_tiny_backend(mode="external_qformer", *, alternate=True, fingertip_mode=None, contact=False, pose=False, action_only=False):
    cfg = LatentWorldPolicyConfig(
        latent_action_head_type=mode,
        action_expert_only=action_only,
        future_action_window_size=3,
        action_horizon=4,
        future_prediction=True,
        enable_loss_distill=True,
        lam_encoder_distill_weight=0.1,
        detach_future_feature=True,
        repeated_diffusion_steps=1,
        fingertip_loss_weight=0.05 if fingertip_mode else 0.0,
        fingertip_target_mode=fingertip_mode or "delta",
        flow_cfg=ConditionalFlowMatchingConfig(
            pose_conditioning=pose,
            pose_position_mean=(0.1, 0.2, 1.3),
            pose_position_std=(0.05, 0.06, 0.1),
            pose_attention_heads=4,
            contact_conditioning=contact,
            contact_every_n_blocks=2,
            action_dim=44,
            state_dim=46,
            vlm_dim=32,
            vision_dim=8,
            num_vision_tokens=4,
            hidden_dim=32,
            num_layers=2,
            attention_heads=4,
            horizon_sec=0.15,
            use_alternate_vldit=alternate,
            num_inference_steps=2,
        ),
    )
    processor = TinyProcessor()
    with patch(
        "starVLA.model.framework.vlas.lawam._load_vlm_and_processor",
        return_value=(TinyVLM(), processor, None, 99, None),
    ), patch("starVLA.model.framework.vlas.lawam.load_latent_action_model", return_value=TinyLAM()):
        return LatentWorldPolicyBackend(cfg, vlm_model_id="unused")


def raw_sample():
    return {
        "primary_videos": torch.randint(0, 256, (1, 2, 3, 8, 8), dtype=torch.uint8),
        "wrist_images": torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8),
        "lang": "take a photograph",
        "state": torch.randn(1, 46),
        "action": torch.randn(4, 44),
        "embodiment_id": 28,
        "action_hz": 30.0,
    }


def collate(backend, samples):
    cfg = OmegaConf.load(CONFIG_PATH)
    collator = _build_latent_world_collator(cfg, policy_cfg=backend.model_cfg, training=True)
    collator._processor = backend.processor
    collator._placeholder_token_id = 99
    return collator(samples)


def wrap_backend(backend):
    model = nn.Module()
    model.policy_backend = backend
    model.policy_cfg = backend.model_cfg
    model.config = OmegaConf.load(CONFIG_PATH)
    return model


class ExternalLatentActionHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_padding_is_ignored_and_only_valid_tokens_receive_gradients(self):
        head = LatentActionHead(vlm_hidden_dim=16, lam_code_dim=8)
        context = torch.randn(2, 6, 16, requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]])
        output = head(context, mask)
        altered = context.detach().clone()
        altered[~mask.bool()] = torch.randn_like(altered[~mask.bool()]) * 1000
        torch.testing.assert_close(output, head(altered, mask))
        self.assertEqual(output.shape, (2, 1, 8))
        torch.nn.functional.mse_loss(output, torch.randn_like(output)).backward()
        self.assertEqual(context.grad[~mask.bool()].abs().sum().item(), 0)
        self.assertGreater(context.grad[mask.bool()].abs().sum().item(), 0)
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)

    def test_invalid_masks_fail_without_producing_nan_latents(self):
        head = LatentActionHead(vlm_hidden_dim=16, lam_code_dim=8)
        with self.assertRaisesRegex(ValueError, "valid token"):
            head(torch.randn(2, 6, 16), torch.zeros(2, 6))
        with self.assertRaisesRegex(ValueError, "does not match"):
            head(torch.randn(2, 6, 16), torch.ones(2, 5))

    def test_config_and_train_infer_prompts_agree_in_both_modes(self):
        for mode, count in [("external_qformer", 0), ("vlm_queries", 8)]:
            with self.subTest(mode=mode):
                backend = build_tiny_backend(mode)
                sample = raw_sample()
                train_batch = collate(backend, [sample])
                adapter = LatentWorldPolicyVLMAdapter(
                    model=backend.vlm,
                    processor=backend.processor,
                    config=OmegaConf.load(CONFIG_PATH),
                    placeholder_token="<ACT_PH>",
                    act_queries=backend.num_action_queries,
                    flow_queries=backend.flow_action_num_queries,
                )
                builder = LatentWorldPolicyInferBatchBuilder(
                    policy_cfg=backend.model_cfg, policy_backend=backend, policy_vlm_adapter=adapter,
                )
                infer_batch = builder.build_infer_batch([{
                    "lang": sample["lang"],
                    "primary_image": [sample["primary_videos"][0, 0].permute(1, 2, 0).numpy()],
                    "wrist_image": [im.permute(1, 2, 0).numpy() for im in sample["wrist_images"]],
                    "state": sample["state"].numpy(),
                    "embodiment_id": 28,
                    "action_hz": 30,
                }])
                self.assertEqual(train_batch["act_placeholder_mask"].sum().item(), count)
                self.assertEqual(infer_batch["act_placeholder_mask"].sum().item(), count)
                self.assertEqual(train_batch["flow_placeholder_mask"].sum().item(), 8)
                torch.testing.assert_close(train_batch["input_ids"], infer_batch["input_ids"])
                if count == 0:
                    self.assertIsNone(backend.act_query)

    def test_external_latent_reads_wrist_and_future_targets_do_not_leak(self):
        backend = build_tiny_backend()
        batch = collate(backend, [raw_sample()])
        original = backend._run_shared_encoding_train(
            prepared_batch=batch, source="test", lam_features_with_no_grad=True,
        )
        altered = dict(batch)
        altered["input_ids"] = batch["input_ids"].clone()
        # The third image token represents the right wrist, before the flow queries.
        altered["input_ids"][batch["input_ids"] == 3] = 20
        wrist_changed = backend._run_shared_encoding_train(
            prepared_batch=altered, source="test", lam_features_with_no_grad=True,
        )
        self.assertFalse(torch.allclose(original.pred_action_emb, wrist_changed.pred_action_emb))

        altered = dict(batch)
        altered["primary_video"] = batch["primary_video"].clone()
        altered["primary_video"][:, 1] += 10
        future_changed = backend._run_shared_encoding_train(
            prepared_batch=altered, source="test", lam_features_with_no_grad=True,
        )
        torch.testing.assert_close(original.pred_action_emb, future_changed.pred_action_emb)
        torch.testing.assert_close(original.h_t1_pred, future_changed.h_t1_pred)
        self.assertFalse(torch.allclose(original.h_t1_gt, future_changed.h_t1_gt))

    def test_training_and_inference_use_predicted_future_and_backpropagate(self):
        for mode in ["external_qformer", "vlm_queries"]:
            with self.subTest(mode=mode):
                backend = build_tiny_backend(mode)
                batch = collate(backend, [raw_sample(), raw_sample()])
                decoded = []
                flow_inputs = []
                decoder_hook = backend.lam.decoder.register_forward_hook(
                    lambda module, args, output: decoded.append(output)
                )
                flow_hook = backend.flow.register_forward_pre_hook(
                    lambda module, args, kwargs: flow_inputs.append(kwargs), with_kwargs=True,
                )
                losses = backend(batch=batch)
                torch.testing.assert_close(flow_inputs[0]["h_t1_star"], decoded[0])
                self.assertFalse(flow_inputs[0]["h_t1_star"].requires_grad)
                losses["loss_total"].backward()
                for name, parameter in backend.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(backend.vlm_to_lam.query.grad.abs().sum().item(), 0)
                self.assertGreater(backend.vlm.model.projection.weight.grad.abs().sum().item(), 0)
                decoder_hook.remove()
                flow_hook.remove()
                backend.eval()
                infer_batch = dict(batch)
                infer_batch["primary_image"] = batch["primary_video"][:, 0]
                actions = backend.predict_action(batch=infer_batch, num_inference_steps=2)
                self.assertEqual(actions.shape, (2, 4, 44))
                self.assertTrue(torch.isfinite(actions).all())

    def test_retired_conditioning_fields_are_rejected_and_depth_config_builds(self):
        for config_type in (LatentWorldPolicyConfig, ConditionalFlowMatchingConfig):
            self.assertFalse(any(field.name.startswith("geometry_") for field in fields(config_type)))
        for location, key in (("action_model", "geometry_conditioning"), ("flow_cfg", "geometry_attention_every_n_blocks")):
            cfg = OmegaConf.load(CONFIG_PATH)
            target = cfg.framework.action_model
            if location == "flow_cfg":
                target = target.flow_cfg
            target[key] = True if key.endswith("conditioning") else 4
            with self.assertRaisesRegex(ValueError, "Unknown.*" + key):
                LatentWorldPolicyConfigBuilder(cfg).build()
        cfg = OmegaConf.load(CONFIG_PATH.with_name("starvla_train_dexjoco_bimanual_photograph_C_h30_depth_queries.yaml"))
        self.assertTrue(LatentWorldPolicyConfigBuilder(cfg).build().depth_query_conditioning)

    def test_plain_dit_and_fingertip_auxiliary_modes_still_work(self):
        for mode in (None, "delta", "endpoint", "latent_delta"):
            with self.subTest(fingertip_mode=mode):
                backend = build_tiny_backend(alternate=False, fingertip_mode=mode)
                self.assertFalse(any("geometry" in key for key in backend.state_dict()))
                sample = raw_sample()
                if mode:
                    sample["fingertip_positions"] = torch.randn(4, 8, 3) * 0.01
                    sample["fingertip_valid"] = torch.ones(4, 8, dtype=torch.bool)
                batch = collate(backend, [sample])
                self.assertFalse(any(key.startswith("geometry_") for key in batch))
                losses = backend(batch=batch)
                losses["loss_total"].backward()
                if mode:
                    self.assertGreater(backend.fingertip_head[-1].weight.grad.abs().sum().item(), 0)
                losses["loss_depth"] = torch.tensor(0.25)
                mapped = map_policy_train_output(losses)
                self.assertEqual(mapped["loss_depth"].item(), 0.25)
                self.assertFalse(any("geometry" in key for key in mapped))
                backend.eval()
                infer_batch = dict(batch, primary_image=batch["primary_video"][:, 0])
                actions = backend.predict_action(batch=infer_batch, guidance_scale=1.5)
                self.assertEqual(actions.shape, (1, 4, 44))
                self.assertTrue(torch.isfinite(actions).all())

    def test_head_modes_are_validated_before_loading_models(self):
        cfg = OmegaConf.load(CONFIG_PATH)
        policy_cfg = LatentWorldPolicyConfigBuilder(cfg).build()
        self.assertEqual(policy_cfg.num_vlm_action_queries, 0)
        cfg.framework.action_model.latent_action_head_type = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown latent_action_head_type"):
            LatentWorldPolicyConfigBuilder(cfg).build()
        cfg.framework.action_model.latent_action_head_type = "external_qformer"
        cfg.framework.action_model.flow_only_mode = True
        with self.assertRaisesRegex(ValueError, "flow_only_mode=false"):
            LatentWorldPolicyConfigBuilder(cfg).build()

    def test_external_checkpoints_are_complete_without_vlm_latent_queries(self):
        backend = build_tiny_backend()
        keys = set(wrap_backend(backend).state_dict())
        counts = validate_full_checkpoint_state_dict(
            checkpoint_keys=keys, framework_name="LaWAM", checkpoint_path="external.pt",
            require_vlm_action_queries=False,
        )
        self.assertEqual(counts["policy_backend.act_query"], 0)
        self.assertEqual(counts["policy_backend.vlm_to_lam."], len(backend.vlm_to_lam.state_dict()))
        with self.assertRaisesRegex(RuntimeError, "act_query"):
            validate_full_checkpoint_state_dict(
                checkpoint_keys=keys, framework_name="LaWAM", checkpoint_path="legacy.pt",
            )
        with self.assertRaisesRegex(RuntimeError, "vlm_to_lam"):
            validate_full_checkpoint_state_dict(
                checkpoint_keys={k for k in keys if not k.startswith("policy_backend.vlm_to_lam.")},
                framework_name="LaWAM", checkpoint_path="incomplete.pt", require_vlm_action_queries=False,
            )

    def test_pretrained_head_is_loaded_without_overwriting_stage1_lam(self):
        model = wrap_backend(build_tiny_backend())
        checkpoint = {k: v.detach().clone() + 0.25 for k, v in model.state_dict().items()}
        checkpoint["policy_backend.act_query"] = torch.randn(8, 32)
        lam_before = copy.deepcopy(model.policy_backend.lam.state_dict())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pretrain.pt"
            torch.save(checkpoint, path)
            TrainerUtils.load_finetune_init_weights(model, str(path), load_pretrained_policy_lam=False)
        for key, value in model.policy_backend.vlm_to_lam.state_dict().items():
            torch.testing.assert_close(value, checkpoint["policy_backend.vlm_to_lam." + key])
        for key, value in model.policy_backend.lam.state_dict().items():
            torch.testing.assert_close(value, lam_before[key])
        groups = build_param_lr_groups(model, model.config)
        head_ids = {id(p) for p in model.policy_backend.vlm_to_lam.parameters()}
        head_group = next(group for group in groups if head_ids.intersection(map(id, group["params"])))
        self.assertEqual(head_group["lr"], 5e-5)
        self.assertTrue(head_ids.issubset(set(map(id, head_group["params"]))))

    def test_saved_external_checkpoint_reloads_through_serving_entrypoint(self):
        source = wrap_backend(build_tiny_backend())
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir()
            path = checkpoint_dir / "pytorch_model.pt"
            torch.save(source.state_dict(), path)
            OmegaConf.save(source.config, run_dir / "config.yaml")
            (run_dir / "dataset_statistics.json").write_text(json.dumps({"dexjoco_bimanual": {}}))
            target = wrap_backend(build_tiny_backend())
            with patch("starVLA.model.framework.build_framework", return_value=target):
                restored = baseframework.from_pretrained(str(path))
        self.assertIsNone(restored.policy_backend.act_query)
        self.assertEqual(restored.config.framework.action_model.latent_action_head_type, "external_qformer")
        for key, value in source.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])
        self.assertIn("dexjoco_bimanual", restored.norm_stats)


if __name__ == "__main__":
    unittest.main()
