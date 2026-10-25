"""Planning-pose behavior tests with production heads and tiny CPU VLM/LAM fixtures."""
import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.test_external_latent_action_head import build_tiny_backend, collate, raw_sample, wrap_backend
from scripts.test_pose_action import pose, infer_batch
from starVLA.model.framework.base_framework import baseframework, validate_pose_checkpoint_keys
from starVLA.model.framework.latent_world.batch_builder import LatentWorldPolicyInferBatchBuilder
from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
from starVLA.model.framework.latent_world.vlm_adapter import LatentWorldPolicyVLMAdapter
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from scripts.configure_pose_plan import configure


def batch_for(model):
    samples = [raw_sample(), raw_sample()]
    for sample, value in zip(samples, pose(2)):
        sample['object_pose_condition'] = value
        sample['fingertip_positions'] = torch.randn(4, 8, 3) * .01
        sample['fingertip_valid'] = torch.ones(4, 8, dtype=torch.bool)
    return collate(model, samples)


class PlanningPoseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(37)

    def test_initialization_equivalence_both_heads_and_dits(self):
        for mode in ('vlm_queries', 'external_qformer'):
            for alternate in (False, True):
                torch.manual_seed(3)
                baseline = build_tiny_backend(mode, alternate=alternate, fingertip_mode='endpoint')
                torch.manual_seed(3)
                green = build_tiny_backend(mode, alternate=alternate, fingertip_mode='endpoint', plan_pose=True)
                for key, value in baseline.state_dict().items():
                    torch.testing.assert_close(value, green.state_dict()[key], atol=0, rtol=0)
                batch = batch_for(green)
                for model in (baseline, green):
                    model.train()
                torch.manual_seed(5)
                expected = baseline(batch=batch)
                torch.manual_seed(5)
                actual = green(batch=batch)
                for key in expected:
                    torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)
                for model in (baseline, green):
                    model.eval()
                for guidance in (1., 1.5):
                    torch.manual_seed(9)
                    a = baseline.predict_action(batch=infer_batch(batch), guidance_scale=guidance)
                    torch.manual_seed(9)
                    b = green.predict_action(batch=infer_batch(batch), guidance_scale=guidance)
                    torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_gradients_and_causal_path_with_shared_features_unchanged(self):
        for mode in ('vlm_queries', 'external_qformer'):
            # AlternateVLDiT's first cross-attention is text; layer 2 is the first
            # image/future attention. Two-layer fixtures cannot observe a green path.
            model = build_tiny_backend(mode, plan_pose=True, fingertip_mode='endpoint', layers=4)
            model.model_cfg.detach_future_feature = True
            model.requires_grad_(False)
            model.plan_pose_encoder.requires_grad_(True)
            model.vlm_to_lam.pose_adapter.requires_grad_(True)
            model.train()
            batch = batch_for(model)
            for key in ('loss_distill', 'loss_perceptual', 'loss_fingertip'):
                model.zero_grad(set_to_none=True)
                losses = model(batch=batch)
                losses[key].backward()
                grad = model.vlm_to_lam.pose_adapter.output_projection.weight.grad
                self.assertGreater(grad.abs().sum().item(), 0, (mode, key))
            self.assertFalse(model(batch=batch)['loss_flow'].requires_grad)
            frozen = {k:p.detach().clone() for k,p in model.named_parameters() if not p.requires_grad}
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.003)
            for step in range(4):
                opt.zero_grad(set_to_none=True)
                losses = model(batch=batch)
                self.assertTrue(torch.isfinite(losses['loss_total']))
                losses['loss_total'].backward()
                if step == 1:
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in model.plan_pose_encoder.parameters()
                                           if p.grad is not None), 0)
                opt.step()
            model.eval()
            alternate = copy.deepcopy(batch)
            alternate['object_pose_condition'][:, :3] += .2
            def encode(b):
                return model._run_shared_encoding_train(prepared_batch=b, source='test', lam_features_with_no_grad=True)
            a, b = encode(batch), encode(alternate)
            for key in ('h_vlm', 'h_t', 'h_t1_gt'):
                torch.testing.assert_close(getattr(a,key), getattr(b,key), atol=0, rtol=0)
            for key in ('pred_action_emb', 'h_t1_pred'):
                self.assertGreater((getattr(a,key)-getattr(b,key)).abs().max().item(), 0)
            for k,p in model.named_parameters():
                if k in frozen:
                    torch.testing.assert_close(p,frozen[k],atol=0,rtol=0)
            for guidance in (1., 1.5):
                torch.manual_seed(10)
                aa = model.predict_action(batch=infer_batch(batch),guidance_scale=guidance)
                torch.manual_seed(10)
                bb = model.predict_action(batch=infer_batch(alternate),guidance_scale=guidance)
                self.assertGreater((aa-bb).abs().max().item(), 0)
            endpoint_a = model.fingertip_head(a.h_t1_pred.float().mean(1))
            endpoint_b = model.fingertip_head(b.h_t1_pred.float().mean(1))
            self.assertGreater((endpoint_a-endpoint_b).abs().max().item(), 0)
            inferred = model._run_shared_encoding_infer(prepared_batch=infer_batch(batch), source='test', lam_features_with_no_grad=True)
            torch.testing.assert_close(a.pred_action_emb,inferred.pred_action_emb,atol=0,rtol=0)

    def test_pose_example_batch_and_missing_field(self):
        model = build_tiny_backend('vlm_queries', plan_pose=True)
        builder = LatentWorldPolicyInferBatchBuilder(
            policy_cfg=model.model_cfg, policy_backend=model,
            policy_vlm_adapter=LatentWorldPolicyVLMAdapter(model=model.vlm,processor=model.processor,
                config=OmegaConf.load('starVLA/config/training/starvla_train_dexjoco_photograph_pose_plan_stage2.yaml'),
                placeholder_token='<ACT_PH>',act_queries=8,flow_queries=8))
        example = dict(primary_image=[np.zeros((8,8,3),np.uint8)], lang='take a photograph',
                       state=np.zeros(46), action_hz=30., embodiment_id=28,object_pose_condition=pose()[0].numpy())
        batch = builder.build_infer_batch([example])
        np.testing.assert_array_equal(batch['object_pose_condition'][0].numpy(), example['object_pose_condition'])
        with self.assertRaises(KeyError):
            builder.build_infer_batch([{k:v for k,v in example.items() if k!='object_pose_condition'}])
        batch = batch_for(model)
        del batch['object_pose_condition']
        with self.assertRaises(KeyError):
            model(batch=batch)
        with self.assertRaises(KeyError):
            collate(model,[raw_sample()])
        wrapped = wrap_backend(model)
        config = asdict(model.model_cfg);config.pop('vlm_dtype')
        wrapped.config.framework.action_model = OmegaConf.create(config)
        wrapped.config.framework.action_model.action_expert_only = True
        with self.assertRaisesRegex(ValueError,'action_expert_only=false'):
            LatentWorldPolicyConfigBuilder(wrapped.config).build()

    def test_green_both_switches_and_checkpoint_roundtrip(self):
        for orange in (False, True):
            model = wrap_backend(build_tiny_backend('vlm_queries', plan_pose=True, pose=orange, fingertip_mode='endpoint'))
            cfg = asdict(model.policy_cfg)
            cfg.pop('vlm_dtype')
            model.config.framework.action_model = OmegaConf.create(cfg)
            built = LatentWorldPolicyConfigBuilder(model.config).build()
            self.assertTrue(built.plan_pose_conditioning)
            self.assertEqual(built.flow_cfg.pose_conditioning, orange)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory);(root/'checkpoints').mkdir()
                checkpoint = root/'checkpoints/steps_4_pytorch_model.pt'
                torch.save(model.state_dict(),checkpoint)
                OmegaConf.save(model.config,root/'config.yaml')
                (root/'dataset_statistics.json').write_text(json.dumps({'dexjoco_bimanual':{}}))
                restored = wrap_backend(build_tiny_backend('vlm_queries',plan_pose=True,pose=orange,fingertip_mode='endpoint'))
                restored.config = model.config
                with patch('starVLA.model.framework.build_framework',return_value=restored):
                    loaded = baseframework.from_pretrained(str(checkpoint))
                for key,value in model.state_dict().items():
                    torch.testing.assert_close(value,loaded.state_dict()[key],atol=0,rtol=0)
                keys = set(model.state_dict())
                without_plan = {k for k in keys if not k.startswith(('policy_backend.plan_pose_encoder.',
                                                                   'policy_backend.vlm_to_lam.pose_adapter.'))}
                for expected,actual in ((keys,without_plan),(without_plan,keys)):
                    with self.assertRaises(RuntimeError):
                        validate_pose_checkpoint_keys(expected,actual)
        # Baseline-to-green initialization is permitted in training and preserves old tensors.
        source = wrap_backend(build_tiny_backend('vlm_queries'))
        target = wrap_backend(build_tiny_backend('vlm_queries',plan_pose=True))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'baseline.pt';torch.save(source.state_dict(),checkpoint)
            TrainerUtils.load_finetune_init_weights(target,checkpoint_path=str(checkpoint))
            for key,value in source.state_dict().items():
                torch.testing.assert_close(value,target.state_dict()[key],atol=0,rtol=0)

    def test_relocated_config_assets_statistics_and_missing_video(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/'new_data/bimanual_photograph'
            for folder in ('data', 'meta/episodes', 'videos'):
                (data/folder).mkdir(parents=True)
            (data/'meta/info.json').write_text('{"fps":30}')
            episodes = list(range(87))
            (data/'pose_train_episode_indices.json').write_text(json.dumps(episodes))
            (data/'object_pose_provenance.json').write_text('{}')
            (data/'videos/example.mp4').write_bytes(b'fixture')
            values = np.zeros((87,12),np.float32);values[:,[3,7,11]]=1
            values[:,:3] = np.arange(87)[:,None]
            pq.write_table(pa.table({'episode_index':episodes,'object_pose_condition':list(values),
                                    'object_pose_valid':[True]*87}), data/'data/rows.parquet')
            pq.write_table(pa.table({'episode_index':episodes}),data/'meta/episodes/episodes.parquet')
            side = root/'new_tips/bimanual_photograph';side.mkdir(parents=True)
            (side/'all.npz').write_bytes(b'fixture')
            (side/'manifest.json').write_text(json.dumps({'episodes':[
                {'episode_index':ep,'file':'all.npz'} for ep in episodes]}))
            assets = root/'new_assets'
            for relative in ('results/Checkpoints/qwen3_weights/config.json',
                             'results/Checkpoints/dinov3-vitb16-pretrain-lvd1689m/config.json',
                             'results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt',
                             'latent_action_model/logs/dino_large_vae_dexjoco_bimanual_exclude_photograph_fingertip_stage1/pytorch_model.pt'):
                path = assets/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
            lam = assets/'latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml'
            lam.parent.mkdir(parents=True);lam.write_text('model:\n  vision_model_id: /old/machine/dino\n')
            cfg = configure(data_root=data,fingertip_root=side.parent,asset_root=assets,
                            output_root=root/'runs',output=root/'config/green.yaml',steps=20)
            self.assertEqual(cfg.framework.action_model.plan_pose_position_mean,[42.5]*3)
            self.assertEqual(cfg.pose_data_provenance.validation_episodes,[86])
            self.assertTrue(str(cfg.framework.qwenvl.base_vlm).startswith(str(assets)))
            relocated_lam = OmegaConf.load(cfg.framework.action_model.lam_yaml_path)
            self.assertEqual(relocated_lam.model.vision_model_id,str(assets/'results/Checkpoints/dinov3-vitb16-pretrain-lvd1689m'))
            self.assertIn('/old/machine/dino',lam.read_text())
            self.assertTrue(cfg.trainer.load_pretrained_policy_flow)
            self.assertFalse(cfg.trainer.load_pretrained_policy_lam)
            self.assertFalse(cfg.trainer.use_pretrained_dataset_statistics)
            (data/'videos/broken.mp4').symlink_to('/missing/video.mp4')
            with self.assertRaisesRegex(FileNotFoundError,'Broken dataset video links'):
                configure(data_root=data,fingertip_root=side.parent,asset_root=assets,
                          output_root=root/'runs',output=root/'other.yaml',steps=20)


if __name__ == '__main__':
    unittest.main()
