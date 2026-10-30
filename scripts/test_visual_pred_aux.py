"""Predicted visual tokens: causal inputs, future supervision and action conditioning."""
import copy
import unittest
from unittest.mock import patch

import torch

from scripts.test_external_latent_action_head import build_tiny_backend, collate, raw_sample
from scripts.test_interaction import geometry
from scripts.test_pose_action import infer_batch
from starVLA.model.framework.latent_world.runtime.output_mapper import map_policy_train_output


def make_batch(model):
    samples = [raw_sample(), raw_sample()]
    for sample, value, valid in zip(samples, *geometry()):
        sample['future_interaction_geometry'] = value
        sample['future_interaction_valid'] = valid
    return collate(model, samples)


class PredictedVisualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(35)

    def test_same_architecture_and_initialization_as_current_visual_aux(self):
        for alternate in (False, True):
            torch.manual_seed(5)
            current = build_tiny_backend('vlm_queries', alternate=alternate, interaction='visual_aux')
            torch.manual_seed(5)
            future = build_tiny_backend('vlm_queries', alternate=alternate, interaction='visual_pred_aux')
            self.assertEqual(set(current.state_dict()), set(future.state_dict()))
            for key, value in current.state_dict().items():
                torch.testing.assert_close(future.state_dict()[key], value, atol=0, rtol=0)
            self.assertIsNone(future.wm_geometry_head)

    def test_tokens_read_predicted_future_and_current_context_without_gt_leakage(self):
        model = build_tiny_backend('vlm_queries', interaction='visual_pred_aux').eval()
        batch = make_batch(model)
        seen = []
        hook = model.interaction_encoder.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs), with_kwargs=True)
        encode = lambda b: model._run_shared_encoding_train(
            prepared_batch=b, source='test', lam_features_with_no_grad=True)
        first = encode(batch)
        torch.testing.assert_close(seen[0]['current_vision'], first.h_t1_pred)
        self.assertFalse(torch.allclose(first.h_t_original, seen[0]['current_vision']))
        torch.testing.assert_close(seen[0]['state'], batch['state'])
        self.assertEqual(first.interaction_tokens.shape, (2, 3, 16))
        changed = copy.deepcopy(batch)
        changed['primary_video'][:, -1] += 10
        changed['future_interaction_geometry'].fill_(100)
        changed['interaction_geometry'] = torch.full_like(batch['future_interaction_geometry'], float('nan'))
        changed['interaction_valid'] = torch.ones_like(batch['future_interaction_valid'])
        second = encode(changed)
        torch.testing.assert_close(first.interaction_tokens, second.interaction_tokens, atol=0, rtol=0)
        self.assertFalse(torch.allclose(first.h_t1_gt, second.h_t1_gt))
        with patch.object(model, '_decode_future_tokens_strict_single_query', return_value=first.h_t1_pred + 1):
            third = encode(batch)
        self.assertFalse(torch.allclose(first.interaction_tokens, third.interaction_tokens))
        hook.remove()

    def test_future_full_geometry_loss_and_gradients(self):
        model = build_tiny_backend('vlm_queries', interaction='visual_pred_aux').train()
        self.assertTrue(model.model_cfg.detach_future_feature)
        batch = make_batch(model)
        # Only flange-relative coordinates valid: these must remain supervised,
        # unlike the smaller dynamic-state subset in visual_wm_aux.
        batch['future_interaction_valid'].zero_()
        batch['future_interaction_valid'][..., 9:15] = True
        batch['future_interaction_geometry'][..., 9:15] = .4
        result = model(batch=batch)
        self.assertGreater(result['loss_interaction'].item(), 0)
        result['loss_interaction'].backward()
        for module in (model.interaction_encoder, model.lam.decoder, model.vlm_to_lam, model.vlm):
            grad = sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)
            self.assertGreater(grad, 0)
        self.assertTrue(all(p.grad is None for p in model.flow.parameters()))
        self.assertIn('loss_interaction', map_policy_train_output(result))
        expected = (result['loss_flow'] + model.model_cfg.perceptual_weight * result['loss_perceptual']
                    + model.model_cfg.lam_encoder_distill_weight * result['loss_distill']
                    + model.model_cfg.interaction_loss_weight * result['loss_interaction'])
        torch.testing.assert_close(result['loss_total'], expected)
        changed = copy.deepcopy(batch)
        changed['interaction_geometry'] = torch.full_like(batch['future_interaction_geometry'], 999)
        changed['interaction_valid'] = torch.ones_like(batch['future_interaction_valid'])
        torch.testing.assert_close(model(batch=changed)['loss_interaction'], result['loss_interaction'])
        changed['future_interaction_geometry'][..., 9:15] += 10
        self.assertGreater(model(batch=changed)['loss_interaction'], result['loss_interaction'])
        with self.assertRaisesRegex(KeyError, 'aligned future'):
            model(batch={k: v for k, v in batch.items() if not k.startswith('future_interaction_')})
        with self.assertRaisesRegex(KeyError, 'future_interaction_geometry'):
            collate(model, [raw_sample()])

    def test_padded_future_has_graph_connected_zero_loss(self):
        model = build_tiny_backend('vlm_queries', interaction='visual_pred_aux').train()
        batch = make_batch(model)
        batch['future_interaction_valid'].zero_()
        batch['future_interaction_geometry'].fill_(float('nan'))
        result = model(batch=batch)
        self.assertTrue(torch.isfinite(result['loss_total']))
        self.assertEqual(result['loss_interaction'].item(), 0)
        result['loss_interaction'].backward()
        for module in (model.interaction_encoder, model.lam.decoder):
            for name, param in module.named_parameters():
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_tokens_condition_both_action_experts_and_label_free_inference(self):
        for alternate in (False, True):
            model = build_tiny_backend('vlm_queries', alternate=alternate, interaction='visual_pred_aux')
            model.model_cfg.repeated_diffusion_steps = 2
            batch = make_batch(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            model.train()
            # Open the zero-initialized interaction residuals with an action-only update.
            model(batch=batch)['loss_flow'].backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model(batch=batch)['loss_flow'].backward()
            for module in (model.interaction_encoder, model.lam.decoder):
                self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()
                                       if p.grad is not None), 0)
            model.eval()
            full = infer_batch(batch)
            label_free = {k: v for k, v in full.items() if not k.startswith('future_interaction_')}
            for guidance in (1., 1.5):
                torch.manual_seed(7)
                expected = model.predict_action(batch=full, guidance_scale=guidance)
                torch.manual_seed(7)
                actual = model.predict_action(batch=label_free, guidance_scale=guidance)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                hook = model.interaction_encoder.register_forward_hook(
                    lambda module, inputs, output: (torch.zeros_like(output[0]), output[1]))
                torch.manual_seed(7)
                ablated = model.predict_action(batch=label_free, guidance_scale=guidance)
                hook.remove()
                self.assertFalse(torch.allclose(actual, ablated, atol=1e-6, rtol=0))


if __name__ == '__main__':
    unittest.main()
