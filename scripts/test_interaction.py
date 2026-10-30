"""Behavioral acceptance checks for multi-object conditioning, with tiny production backends."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.test_external_latent_action_head import build_tiny_backend, collate, raw_sample
from scripts.test_pose_action import infer_batch
from scripts.configure_interaction import configure, parser
from starVLA.interaction_geometry import SCHEMA, FEATURE_DIM, FEATURE_NAMES, ENTITY_NAMES, InteractionStore, validate_geometry
from starVLA.interaction_data import audit_store, feature_statistics
from starVLA.interaction_replay import match_source
from starVLA.model.framework.base_framework import validate_pose_checkpoint_keys
from starVLA.model.framework.vlas.interaction_tokens import InteractionTokens


def geometry(batch=2,entities=7):
    value=torch.zeros(batch,entities,FEATURE_DIM)
    value[...,:3]=torch.randn(batch,entities,3)*.1
    value[...,3]=value[...,7]=1
    valid=torch.ones_like(value,dtype=torch.bool)
    valid[...,15]=False
    return value,valid


def batch_for(model):
    samples=[raw_sample(),raw_sample()]
    for sample,value,valid in zip(samples,*geometry()):
        sample['interaction_geometry']=value
        sample['interaction_valid']=valid
    return collate(model,samples)


class InteractionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def test_zero_init_equivalence_train_and_inference_both_dits(self):
        for alternate in (False,True):
            for mode in ('oracle','visual','visual_aux'):
                torch.manual_seed(3)
                baseline=build_tiny_backend('vlm_queries',alternate=alternate)
                torch.manual_seed(3)
                model=build_tiny_backend('vlm_queries',alternate=alternate,interaction=mode)
                for key,value in baseline.state_dict().items():
                    torch.testing.assert_close(value,model.state_dict()[key],atol=0,rtol=0)
                batch=batch_for(model)
                for module in (baseline,model): module.train()
                torch.manual_seed(2)
                expected=baseline(batch=batch)
                torch.manual_seed(2)
                actual=model(batch=batch)
                for key in ('loss_flow','loss_perceptual','loss_distill'):
                    torch.testing.assert_close(actual[key],expected[key],atol=0,rtol=0)
                for module in (baseline,model): module.eval()
                for scale in (1.,1.5):
                    torch.manual_seed(6)
                    expected=baseline.predict_action(batch=infer_batch(batch),guidance_scale=scale)
                    torch.manual_seed(6)
                    actual=model.predict_action(batch=infer_batch(batch),guidance_scale=scale)
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0)

    def test_stage2_gradients_repeat_and_geometry_affects_only_action_path(self):
        for mode in ('oracle','visual','visual_aux'):
            model=build_tiny_backend('vlm_queries',interaction=mode)
            model.model_cfg.repeated_diffusion_steps=2
            batch=batch_for(model)
            model.train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                losses=model(batch=batch)
                self.assertTrue(torch.isfinite(losses['loss_total']))
                losses['loss_total'].backward()
                for name,param in model.interaction_encoder.named_parameters():
                    self.assertIsNotNone(param.grad,(mode,name))
                    self.assertTrue(torch.isfinite(param.grad).all(),(mode,name))
                if step==1:
                    grad=sum(p.grad.abs().sum().item() for p in model.interaction_encoder.parameters())
                    self.assertGreater(grad,0)
                optimizer.step()
            if mode=='oracle':
                model.eval()
                changed=copy.deepcopy(batch)
                changed['interaction_geometry'][...,:3]+=.5
                encode=lambda b:model._run_shared_encoding_train(prepared_batch=b,source='test',lam_features_with_no_grad=True)
                a,b=encode(batch),encode(changed)
                for key in ('h_vlm','h_t','h_t1_pred','pred_action_emb'):
                    torch.testing.assert_close(getattr(a,key),getattr(b,key),atol=0,rtol=0)
                torch.manual_seed(1)
                x=model.predict_action(batch=infer_batch(batch))
                torch.manual_seed(1)
                y=model.predict_action(batch=infer_batch(changed))
                self.assertFalse(torch.allclose(x,y,atol=1e-6,rtol=0))

    def test_visual_aux_cannot_read_labels_at_inference_and_matches_visual_architecture(self):
        torch.manual_seed(7)
        visual=build_tiny_backend('vlm_queries',interaction='visual')
        torch.manual_seed(7)
        aux=build_tiny_backend('vlm_queries',interaction='visual_aux')
        for key,value in visual.state_dict().items():
            torch.testing.assert_close(aux.state_dict()[key],value,atol=0,rtol=0)
        batch=batch_for(aux)
        aux.eval()
        full=infer_batch(batch)
        stripped={k:v for k,v in full.items() if not k.startswith('interaction_')}
        torch.manual_seed(1)
        a=aux.predict_action(batch=full)
        torch.manual_seed(1)
        b=aux.predict_action(batch=stripped)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        aux.train()
        with self.assertRaisesRegex(KeyError,'requires audited'):
            aux(batch={k:v for k,v in batch.items() if not k.startswith('interaction_')})

    def test_oracle_padding_and_vlm_padding_are_ignored(self):
        model=InteractionTokens(mode='oracle',vision_dim=8,vlm_dim=8,state_dim=4,entities=7,dim=8,queries=3,heads=2,layers=1)
        value,valid=geometry()
        valid[:,2]=False
        kwargs=dict(current_vision=torch.randn(2,4,8),vlm=torch.randn(2,5,8),
                    language_mask=torch.tensor([[1,1,1,0,0]]*2).bool(),state=torch.randn(2,4),state_mask=torch.ones(2,4))
        a=model(**kwargs,geometry=value,geometry_mask=valid)
        changed=value.clone()
        changed[:,2]=float('nan')
        kwargs['vlm'][:,3:]=10000
        b=model(**kwargs,geometry=changed,geometry_mask=valid)
        torch.testing.assert_close(a[0],b[0],atol=0,rtol=0)
        with self.assertRaisesRegex(ValueError,'at least one valid'):
            model(**kwargs,geometry=value,geometry_mask=torch.zeros_like(valid))

    def test_checkpoint_mismatch_is_rejected(self):
        keys=set('policy_backend.'+k for k in build_tiny_backend('vlm_queries',interaction='visual').state_dict())
        with self.assertRaises(RuntimeError):
            validate_pose_checkpoint_keys(keys,{k for k in keys if 'interaction_encoder' not in k})

    def test_sidecar_full_alignment_mask_and_train_only_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            task='bimanual_hanoi'
            target=root/task
            (target/'episodes').mkdir(parents=True)
            entries={}
            dataset={}
            for ep in range(3):
                value,valid=geometry(3)
                value[...,:3]=1000 if ep==2 else ep
                arrays=dict(interaction_geometry=value.numpy(),interaction_valid=valid.numpy(),
                            state=np.zeros((3,46),np.float32),action=np.zeros((3,44),np.float32),
                            timestamp=np.arange(3)/30,frame_index=np.arange(3))
                np.savez_compressed(target/'episodes'/f'episode-{ep:06d}.npz',**arrays)
                entries[str(ep)]=dict(length=3,valid=True)
                dataset[ep]=arrays
            meta=dict(schema=SCHEMA,task=task,features=FEATURE_NAMES,entities=ENTITY_NAMES[task],coordinate_frame='world',fps=30,episodes=entries)
            (target/'manifest.json').write_text(json.dumps(meta))
            store=InteractionStore(root)
            audit_store(store,task,dataset,30,[0,1,2])
            stats=feature_statistics(store,task,30,[0,1])
            self.assertAlmostEqual(stats['mean'][0],.5)
            dataset[1]['state'][1,0]=.1
            with self.assertRaisesRegex(ValueError,'state mismatch'):
                audit_store(store,task,dataset,30,[1])
            meta['episodes']['0']['valid']=False
            (target/'manifest.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'Missing/invalid'):
                InteractionStore(root).episode(task,0,30)
            value,valid=geometry(1)
            value[...,6:9]=value[...,3:6]
            with self.assertRaisesRegex(ValueError,'orthogonal'):
                validate_geometry(value.numpy(),valid.numpy(),task)

    def test_raw_mapping_uses_entire_trajectory_and_correct_prefix(self):
        state=np.arange(5*46).reshape(5,46).astype(np.float32)
        action=np.arange(5*44).reshape(5,44).astype(np.float32)
        dataset={9:dict(state=state[2:].copy(),action=action[2:].copy())}
        self.assertEqual(match_source(state,action,dataset),[(9,2)])
        dataset[9]['state'][1,0]+=.1
        self.assertEqual(match_source(state,action,dataset),[])

    def test_four_configs_same_cohort_h30_assets_and_visual_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            task='bimanual_microwave_cook'
            data=root/'data'/task
            (data/'meta/episodes').mkdir(parents=True)
            (data/'data').mkdir()
            (data/'meta/info.json').write_text(json.dumps(dict(fps=30)))
            pq.write_table(pa.Table.from_pylist([dict(episode_index=i,length=3) for i in range(3)]),data/'meta/episodes/file.parquet')
            rows=[dict(episode_index=i,frame_index=t,timestamp=t/30,**{'observation.state':[0.]*46,'action':[0.]*44}) for i in range(3) for t in range(3)]
            pq.write_table(pa.Table.from_pylist(rows),data/'data/file.parquet')
            side=root/'geometry'/task
            (side/'episodes').mkdir(parents=True)
            meta=dict(schema=SCHEMA,task=task,features=FEATURE_NAMES,entities=ENTITY_NAMES[task],coordinate_frame='world',fps=30,episodes={})
            for ep in range(3):
                value,valid=geometry(3,6)
                np.savez_compressed(side/'episodes'/f'episode-{ep:06d}.npz',interaction_geometry=value.numpy(),interaction_valid=valid.numpy(),
                                    state=np.zeros((3,46)),action=np.zeros((3,44)),frame_index=np.arange(3),timestamp=np.arange(3)/30)
                meta['episodes'][str(ep)]=dict(length=3,valid=True)
            (side/'manifest.json').write_text(json.dumps(meta))
            asset=root/'assets'
            for p in ('results/Checkpoints/qwen3_weights/config.json','results/Checkpoints/dinov3-vitb16-pretrain-lvd1689m/config.json',
                      'results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt',
                      'latent_action_model/logs/dino_large_vae_dexjoco_bimanual_exclude_photograph_fingertip_stage1/pytorch_model.pt'):
                file=asset/p
                file.parent.mkdir(parents=True,exist_ok=True)
                file.write_text('{}')
            file=asset/'latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml'
            file.parent.mkdir(parents=True)
            file.write_text('model:\n  vision_model_id: placeholder\n')
            args=parser().parse_args(['--task','microwave','--data-root',str(data.parent),'--sidecar-root',str(side.parent),
                                      '--asset-root',str(asset),'--config-root',str(root/'configs'),'--output-root',str(root/'checkpoints')])
            configs=[OmegaConf.load(f) for f in configure(args)]
            self.assertEqual(len({c.interaction_experiment.sha256 for c in configs}),1)
            for c in configs:
                self.assertEqual(c.framework.action_model.action_horizon,30)
                self.assertFalse(c.framework.action_model.action_expert_only)
                self.assertEqual(c.trainer.max_train_steps,40000)
                self.assertEqual(c.datasets.vla_data.per_device_batch_size,4)
            self.assertEqual(configs[2].framework.action_model.interaction_queries,configs[3].framework.action_model.interaction_queries)
            self.assertFalse(configs[2].datasets.vla_data.load_interaction_geometry)
            self.assertTrue(configs[3].datasets.vla_data.load_interaction_geometry)
            from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
            for mode in ('visual_wm_aux', 'visual_wm_aux_detach', 'visual_pred_aux'):
                args.variant = mode
                future = OmegaConf.load(configure(args)[0])
                built = LatentWorldPolicyConfigBuilder(future).build()
                self.assertEqual(future.interaction_experiment.sha256, configs[0].interaction_experiment.sha256)
                self.assertTrue(future.datasets.vla_data.load_future_interaction_geometry)
                self.assertEqual(built.flow_cfg.interaction_conditioning, mode == 'visual_pred_aux')
                if mode == 'visual_pred_aux':
                    self.assertEqual(built.interaction_queries, 8)
                    self.assertEqual(built.fingertip_loss_weight, 0)
                self.assertTrue(built.detach_future_feature)
                self.assertGreater(built.interaction_loss_weight, 0)
                for field, value in (('future_prediction', False), ('action_expert_only', True),
                                     ('flow_future_feature', 'current'),
                                     ('enable_flow_h_t1_scheduled_sampling', True),
                                     ('interaction_loss_weight', 0)):
                    invalid = copy.deepcopy(future)
                    invalid.framework.action_model[field] = value
                    with self.assertRaises(ValueError):
                        LatentWorldPolicyConfigBuilder(invalid).build()
            meta['accepted_episodes'] = [0, 1]
            meta['acceptance_policy'] = 'user_accepted_reconstruction'
            (side/'manifest.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'accepted_episodes.json'):
                configure(args)


if __name__=='__main__':
    unittest.main()
