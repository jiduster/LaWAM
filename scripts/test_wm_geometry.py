"""Checks of future-label alignment, causal gradient isolation and policy invariance."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.test_external_latent_action_head import build_tiny_backend, raw_sample, collate
from scripts.test_pose_action import infer_batch
from scripts.test_interaction import geometry
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.model.framework.vlas.wm_geometry import FutureGeometryHead
from starVLA.model.framework.latent_world.runtime.output_mapper import map_policy_train_output


def make_batch(model):
    samples = [raw_sample(), raw_sample()]
    for sample, value, valid in zip(samples, *geometry()):
        sample['future_interaction_geometry'] = value
        sample['future_interaction_valid'] = valid
    return collate(model, samples)


class FutureGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(27)

    def test_auxiliary_gradient_reaches_world_and_latent_only_when_attached(self):
        for mode in ('visual_wm_aux', 'visual_wm_aux_detach'):
            model = build_tiny_backend('vlm_queries', interaction=mode)
            model.train()
            batch = make_batch(model)
            result = model(batch=batch)
            self.assertTrue(torch.isfinite(result['loss_total']))
            result['loss_wm_geometry'].backward()
            self.assertGreater(sum(p.grad.abs().sum().item() for p in model.wm_geometry_head.parameters() if p.grad is not None), 0)
            for module in (model.lam.decoder, model.vlm_to_lam, model.vlm):
                grad = sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)
                if mode.endswith('_detach'):
                    self.assertEqual(grad, 0)
                else:
                    self.assertGreater(grad, 0)
            self.assertTrue(all(p.grad is None for p in model.flow.parameters()))
            model.zero_grad(set_to_none=True)
            model(batch=batch)['loss_total'].backward()
            # Detach is local to the auxiliary branch; full Stage 2 remains trainable.
            for module in (model.flow, model.lam.decoder, model.vlm):
                self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None), 0)
            self.assertIn('wm_position_rmse_m', map_policy_train_output(result))

    def test_identical_policy_initialization_and_label_free_inference(self):
        models = []
        for mode in ('none', 'visual_wm_aux', 'visual_wm_aux_detach'):
            torch.manual_seed(8)
            models.append(build_tiny_backend('vlm_queries', interaction=mode))
        baseline, attached, detached = models
        for key, value in baseline.state_dict().items():
            for model in (attached, detached):
                torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
        self.assertEqual(set(attached.state_dict()), set(detached.state_dict()))
        for key, value in attached.state_dict().items():
            torch.testing.assert_close(detached.state_dict()[key], value, atol=0, rtol=0)
        batch = make_batch(attached)
        losses = []
        actions = []
        for model in models:
            model.train()
            torch.manual_seed(3)
            losses.append(model(batch=batch))
            model.eval()
            # Inference has no sidecar dependency and never calls the readout.
            def forbidden(*args, **kwargs):
                raise AssertionError('Inference called the training-only readout')
            if model.wm_geometry_head is not None:
                model.wm_geometry_head.register_forward_hook(forbidden)
            torch.manual_seed(4)
            label_free = {k: v for k, v in infer_batch(batch).items() if not k.startswith('future_interaction_')}
            actions.append(model.predict_action(batch=label_free))
        for key in ('loss_flow', 'loss_perceptual', 'loss_distill'):
            for result in losses[1:]:
                torch.testing.assert_close(result[key], losses[0][key], atol=0, rtol=0)
        torch.testing.assert_close(losses[1]['loss_wm_geometry'], losses[2]['loss_wm_geometry'], atol=0, rtol=0)
        for action in actions[1:]:
            torch.testing.assert_close(action, actions[0], atol=0, rtol=0)

    def test_future_index_comes_from_video_not_hardcoded_horizon_and_masks_tail(self):
        calls = []
        def get_geometry(ep, frame):
            calls.append((ep, frame))
            return np.full((7, 17), frame, dtype=np.float32), np.ones((7, 17), dtype=bool)
        dataset = SimpleNamespace(
            data_cfg={'load_future_interaction_geometry': True}, _interaction_store=object(),
            delta_indices={'video.ego': np.array([0, 29])},
            episodes_hf=[{'length': 100}], _episode_index_to_row={3: 0}, get_interaction_geometry=get_geometry)
        method = LeRobotSingleDataset.get_future_interaction_geometry
        value, valid, index = method(dataset, 3, 10, 'video.ego')
        self.assertEqual(index, 39)
        self.assertTrue(valid.all())
        self.assertEqual(value[0, 0], 39)
        _, valid, index = method(dataset, 3, 90, 'video.ego')
        self.assertEqual(index, 99)
        self.assertFalse(valid.any())
        dataset.delta_indices['video.ego'] = np.array([0, 7])
        self.assertEqual(method(dataset, 3, 10, 'video.ego')[2], 17)

    def test_only_dynamic_states_contribute_and_all_padded_loss_is_safe(self):
        for task, entities in (('bimanual_hanoi', 7), ('bimanual_microwave_cook', 6)):
            head = FutureGeometryHead(vision_dim=8, task=task, dim=16, heads=4, layers=1,
                                      mean=[0.]*17, std=[1.]*17)
            target, valid = geometry(2, entities)
            future = torch.randn(2, 4, 8, requires_grad=True)
            pred = head(future)
            original = head.loss(pred, target, valid)[0]
            changed = target.clone()
            changed[:, ~head.supervision_mask] = 1e5
            torch.testing.assert_close(head.loss(pred, changed, valid)[0], original)
            padded = head.loss(pred, target, torch.zeros_like(valid))[0]
            self.assertEqual(padded.item(), 0.)
            padded.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))


if __name__ == '__main__':
    unittest.main()
