"""Checks for the joint current/predicted compact-token auxiliary mode."""
import copy
import unittest

import torch

from scripts.test_external_latent_action_head import build_tiny_backend
from scripts.test_interaction import geometry


class JointVisualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)

    def test_both_frames_change_tokens_and_labels_do_not_enter_encoder(self):
        model = build_tiny_backend('vlm_queries', interaction='visual_joint_aux').eval()
        encoder = model.interaction_encoder
        current = torch.randn(2, 4, 8)
        future = torch.randn(2, 4, 8)
        vlm = torch.randn(2, 3, 32)
        mask = torch.ones(2, 3, dtype=torch.bool)
        state = torch.randn(2, 46)
        state_mask = torch.ones_like(state)
        target, valid = geometry(2)
        kwargs = dict(current_vision=current, future_vision=future, vlm=vlm,
                      language_mask=mask, state=state, state_mask=state_mask,
                      geometry=target, geometry_mask=valid)
        first = encoder(**kwargs)[0]
        changed_current = copy.deepcopy(kwargs)
        changed_current['current_vision'] = current + 1
        self.assertFalse(torch.allclose(first, encoder(**changed_current)[0]))
        changed_future = copy.deepcopy(kwargs)
        changed_future['future_vision'] = future + 1
        self.assertFalse(torch.allclose(first, encoder(**changed_future)[0]))
        changed_labels = copy.deepcopy(kwargs)
        changed_labels['geometry'] = target + 100
        torch.testing.assert_close(first, encoder(**changed_labels)[0])

    def test_future_prediction_is_required(self):
        model = build_tiny_backend('vlm_queries', interaction='visual_joint_aux')
        with self.assertRaisesRegex(KeyError, 'visual_joint_aux requires'):
            model.interaction_encoder(
                current_vision=torch.randn(2, 4, 8), future_vision=None,
                vlm=torch.randn(2, 3, 32), language_mask=torch.ones(2, 3, dtype=torch.bool),
                state=torch.randn(2, 46), state_mask=torch.ones(2, 46))


if __name__ == '__main__':
    unittest.main()
