"""CPU acceptance checks with tiny VLM/LAM substitutes and the production flow/DiT."""
import copy
import json
from dataclasses import asdict
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.test_external_latent_action_head import build_tiny_backend, raw_sample, collate, wrap_backend
from starVLA.model.framework.latent_world.batch_builder import LatentWorldPolicyInferBatchBuilder
from starVLA.model.framework.latent_world.vlm_adapter import LatentWorldPolicyVLMAdapter
from starVLA.model.framework.latent_world.runtime.runner import LatentWorldPolicyRunner
from starVLA.model.framework.vlas.pose_conditioning import PoseEncoder
from starVLA.model.framework.base_framework import baseframework, validate_pose_checkpoint_keys
from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.object_pose import SCHEMA, ObjectPoseStore
from scripts.prepare_photograph_object_pose import prepare
from scripts.configure_pose_action import training_pose_statistics


def pose(batch=1):
    value = torch.zeros(batch, 12)
    value[:, :3] = torch.randn(batch, 3) * .1
    value[:, 3] = value[:, 7] = value[:, 11] = 1
    return value


def make_batch(backend, count=2):
    samples = [raw_sample() for _ in range(count)]
    for sample, value in zip(samples, pose(count)):
        sample['object_pose_condition'] = value
    return collate(backend, samples)


def infer_batch(batch):
    result = {k: v for k,v in batch.items() if k not in ('primary_video','actions','actions_mask')}
    result['primary_image'] = batch['primary_video'][:, 0]
    return result


class PoseActionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        torch.set_num_threads(1)

    def test_zero_initialization_both_dits_cfg_train_and_sample(self):
        for alternate in (False, True):
            torch.manual_seed(1)
            baseline = build_tiny_backend(mode='vlm_queries', alternate=alternate, action_only=True).eval()
            torch.manual_seed(1)
            orange = build_tiny_backend(mode='vlm_queries', alternate=alternate, pose=True, action_only=True).eval()
            # Branch construction also preserves the shared parameter RNG stream.
            for name, value in baseline.state_dict().items():
                torch.testing.assert_close(value, orange.state_dict()[name], rtol=0, atol=0)
            batch = make_batch(orange)
            for scale in (1., 1.5):
                torch.manual_seed(9)
                expected = baseline.predict_action(batch=infer_batch(batch), guidance_scale=scale)
                torch.manual_seed(9)
                actual = orange.predict_action(batch=infer_batch(batch), guidance_scale=scale)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for model in (baseline, orange):
                model.train()
            torch.manual_seed(9)
            expected = baseline(batch=batch)['loss_flow']
            torch.manual_seed(9)
            actual = orange(batch=batch)['loss_flow']
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            altered = copy.deepcopy(infer_batch(batch))
            altered['object_pose_condition'] = pose(2)
            torch.manual_seed(8)
            a = baseline.predict_action(batch=infer_batch(batch))
            torch.manual_seed(8)
            b = baseline.predict_action(batch=altered)
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_short_training_gradients_frozen_features_pose_effect_and_checkpoint(self):
        for alternate in (False, True):
            model = build_tiny_backend(mode='vlm_queries', alternate=alternate, pose=True,
                                       action_only=True, fingertip_mode='endpoint')
            model.model_cfg.repeated_diffusion_steps = 2
            batch = make_batch(model)
            before = {n:p.detach().clone() for n,p in model.named_parameters() if not n.startswith('flow.')}
            self.assertTrue(all(p.requires_grad == n.startswith('flow.') for n,p in model.named_parameters()))
            frozen = [model.vlm,model.lam,model.vlm_to_lam,model.fingertip_head]
            model.train()
            self.assertTrue(all(not m.training for m in frozen))
            shared_before = model._run_shared_encoding_train(prepared_batch=batch, source='test', lam_features_with_no_grad=True)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.003)
            first_projection_grad = 0
            for step in range(5):
                optimizer.zero_grad(set_to_none=True)
                result = model(batch=batch)
                self.assertTrue(torch.isfinite(result['loss_flow']))
                torch.testing.assert_close(result['loss_flow'], result['loss_total'])
                result['loss_flow'].backward()
                if step == 0:
                    first_projection_grad = sum(p.grad.abs().sum().item() for n,p in model.flow.named_parameters()
                                                if 'pose_attention' in n and 'out_proj' in n)
                    self.assertGreater(first_projection_grad,0)
                if step == 1:
                    self.assertGreater(model.flow.pose_encoder.groups[0][0].weight.grad.abs().sum().item(),0)
                optimizer.step()
            shared_after = model._run_shared_encoding_train(prepared_batch=batch, source='test', lam_features_with_no_grad=True)
            for field in ('h_t','h_t1_pred','h_vlm','pred_action_emb'):
                torch.testing.assert_close(getattr(shared_before,field),getattr(shared_after,field),atol=0,rtol=0)
            for n,p in model.named_parameters():
                if n in before:
                    torch.testing.assert_close(p,before[n],atol=0,rtol=0)
                    self.assertIsNone(p.grad)
            model.eval()
            original = infer_batch(batch)
            altered = copy.deepcopy(original)
            altered['object_pose_condition'][:,0] += .2
            seen = []
            hook = model.flow.DiT.register_forward_pre_hook(lambda module,args,kwargs: seen.append(kwargs['pose_tokens'].detach().clone()), with_kwargs=True)
            torch.manual_seed(9)
            a = model.predict_action(batch=original, guidance_scale=1.5)
            hook.remove()
            self.assertEqual(len(seen),4)  # 2 integration steps x cond/uncond
            self.assertTrue(all(torch.equal(seen[0], value) for value in seen))
            torch.manual_seed(9)
            b = model.predict_action(batch=altered, guidance_scale=1.5)
            delta = (a-b).abs().max().item()
            self.assertGreater(delta,1e-7)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'pytorch_model.pt'
                torch.save(model.state_dict(), path)
                restored = build_tiny_backend(mode='vlm_queries', alternate=alternate, pose=True,
                                              action_only=True,fingertip_mode='endpoint').eval()
                restored.load_state_dict(torch.load(path,weights_only=True),strict=True)
                torch.manual_seed(9)
                torch.testing.assert_close(a,restored.predict_action(batch=original,guidance_scale=1.5),rtol=0,atol=0)
            print(f'POSE_TRAIN_OK alternate={alternate} steps=5 initial_output_grad={first_projection_grad:.6g} pose_action_delta={delta:.6g}')

    def test_example_processor_batch_training_and_aligned_eval_chain(self):
        model = build_tiny_backend(mode='vlm_queries', pose=True,action_only=True).eval()
        builder = LatentWorldPolicyInferBatchBuilder(policy_cfg=model.model_cfg, policy_backend=model,
                    policy_vlm_adapter=LatentWorldPolicyVLMAdapter(model=model.vlm, processor=model.processor,
                        config=OmegaConf.load("starVLA/config/training/starvla_train_dexjoco_bimanual_photograph_C_h30_contact.yaml"),
                        placeholder_token="<ACT_PH>",act_queries=8,flow_queries=8))
        value = pose()[0].numpy()
        example = dict(primary_image=[np.zeros((8,8,3),np.uint8)],lang='photograph',
                       embodiment_id=28, action_hz=30.,state=np.zeros(46),object_pose_condition=value)
        batch = builder.build_infer_batch([example])
        np.testing.assert_array_equal(batch['object_pose_condition'].numpy()[0],value)
        self.assertEqual(batch['object_pose_condition'].dtype, torch.float32)
        model.predict_action(batch=batch)
        with self.assertRaises(KeyError):
            builder.build_infer_batch([{k:v for k,v in example.items() if k!='object_pose_condition'}])
        runner = LatentWorldPolicyRunner(policy_backend=model,infer_batch_builder=builder)
        runner.infer_step_with_aligned_targets_from_train_batch(make_batch(model))
        missing = infer_batch(make_batch(model)); del missing['object_pose_condition']
        with self.assertRaisesRegex(KeyError,'object_pose_condition'):
            model.predict_action(batch=missing)

    def test_baseline_finetune_init_and_saved_pose_serving_checkpoint(self):
        source = wrap_backend(build_tiny_backend(mode='vlm_queries', action_only=True, fingertip_mode='endpoint'))
        target = wrap_backend(build_tiny_backend(mode='vlm_queries', pose=True, action_only=True, fingertip_mode='endpoint'))
        config = asdict(target.policy_cfg)
        config.pop('vlm_dtype')
        target.config.framework.action_model = OmegaConf.create(config)
        policy = LatentWorldPolicyConfigBuilder(target.config).build()
        self.assertTrue(policy.flow_cfg.pose_conditioning)
        self.assertTrue(policy.action_expert_only)
        self.assertEqual(list(policy.flow_cfg.pose_position_mean),[.1,.2,1.3])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'checkpoints').mkdir()
            baseline_path=root/'baseline.pt'
            torch.save(source.state_dict(),baseline_path)
            TrainerUtils.load_finetune_init_weights(target,checkpoint_path=baseline_path)
            for name,value in source.state_dict().items():
                torch.testing.assert_close(value,target.state_dict()[name],atol=0,rtol=0)
            checkpoint=root/'checkpoints/pytorch_model.pt'
            torch.save(target.state_dict(),checkpoint)
            OmegaConf.save(target.config,root/'config.yaml')
            (root/'dataset_statistics.json').write_text(json.dumps({'dexjoco_bimanual':{}}))
            restored=wrap_backend(build_tiny_backend(mode='vlm_queries',pose=True,action_only=True,fingertip_mode='endpoint'))
            restored.config=target.config
            with patch('starVLA.model.framework.build_framework',return_value=restored):
                loaded=baseframework.from_pretrained(str(checkpoint))
            self.assertTrue(loaded.config.framework.action_model.flow_cfg.pose_conditioning)
            for name,value in target.state_dict().items():
                torch.testing.assert_close(value,loaded.state_dict()[name],atol=0,rtol=0)

    def test_invalid_conditions_and_checkpoint_config_mismatch(self):
        encoder = PoseEncoder(32,16,[0,0,0],[1,1,1])
        self.assertEqual(encoder(pose()).shape,(1,3,32))
        for value in (torch.zeros(2,12), torch.zeros(12), pose()*float('nan'), torch.ones(2,13)):
            with self.assertRaises(ValueError): encoder(value)
        key = 'policy_backend.flow.pose_encoder.position_mean'
        validate_pose_checkpoint_keys({key},{key})
        for expected,actual in (({key},set()),(set(),{key})):
            with self.assertRaises(RuntimeError): validate_pose_checkpoint_keys(expected,actual)

    def test_aligned_sidecars_current_frame_and_train_only_statistics(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); dataset=root/'bimanual_photograph'; recordings=root/'recordings'; out=root/'sidecars'
            (dataset/'data').mkdir(parents=True); (dataset/'meta/episodes').mkdir(parents=True); recordings.mkdir()
            (dataset/'meta/info.json').write_text('{"fps":30}')
            states=np.zeros((6,46),np.float32)
            pq.write_table(pa.table({'episode_index':[0,0,1,1,2,2], 'frame_index':[0,1]*3,
                                    'timestamp':[0.,1/30]*3, 'observation.state':list(states)}),dataset/'data/file.parquet')
            pq.write_table(pa.table({'episode_index':[0,1,2]}),dataset/'meta/episodes/file.parquet')
            for episode in range(3):
                values=pose(2).numpy(); values[:,:3]=episode*100; values[1,:3]+=1
                np.savez(recordings/f'episode-{episode:06d}.npz',schema=SCHEMA,object_pose_condition=values,
                         state=states[:2],timestamp=[0,1/30],frame_index=[0,1])
            prepare(dataset,out,recordings)
            prepare(dataset,out,validate_only=True)
            store=ObjectPoseStore(out)
            self.assertEqual(float(store.frame(dataset.name,0,0,30)[0]),0)
            self.assertEqual(float(store.frame(dataset.name,0,1,30)[0]),1)
            stats=training_pose_statistics(dataset,out,.01)
            self.assertEqual(stats['training_episodes'],[0,1])
            self.assertEqual(stats['validation_episodes'],[2])
            np.testing.assert_allclose(stats['mean'],[50.5]*3)
            # Reject correct-shaped, time-shifted recordings before publishing.
            with np.load(recordings/'episode-000000.npz') as archive:
                data=dict(archive)
            data['timestamp']=data['timestamp']+.01
            np.savez(recordings/'episode-000000.npz',**data)
            with self.assertRaisesRegex(ValueError,'Timestamp mismatch'):
                prepare(dataset,root/'bad',recordings)


if __name__=='__main__':
    unittest.main()
