from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from scripts.diagnose_lawam_teacher_latent import FixedEpisodeWindows, resolve_checkpoint, summarize
from scripts.test_external_latent_action_head import build_tiny_backend, collate, raw_sample
from starVLA.model.framework.latent_world.runtime.offline_diagnostics import (
    diagnose_train_batch,
    masked_error_per_sample,
)


class RecordingFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.noises = []

    def _compute_dtype(self):
        return torch.float32

    def sample_actions_cfg(self, *, h_t1_star, **kwargs):
        noise = torch.randn(h_t1_star.shape[0], 4, 3) * 0.01
        self.noises.append(noise)
        return noise + h_t1_star.mean(dim=(1, 2)).view(-1, 1, 1)


class FakeBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_cfg = SimpleNamespace(future_prediction=True, flow_future_feature='predicted')
        self.flow = RecordingFlow()

    def _prepare_train_batch(self, *, batch):
        return batch

    def _run_shared_encoding_train(self, **kwargs):
        zeros = torch.zeros(2, 4, 3)
        return SimpleNamespace(
            pred_action_emb=torch.zeros(2, 1, 3), h_t=zeros,
            h_t1_pred=zeros, h_t1_gt=zeros + 2, h_vlm=zeros,
        )

    def _run_lam_teacher(self, **kwargs):
        return torch.ones(2, 1, 3)

    def _decode_future_tokens_strict_single_query(self, *, h_t, pred_action_emb, **kwargs):
        return h_t + pred_action_emb


def fake_batch():
    return {
        'primary_video': torch.zeros(2, 2, 3, 4, 4),
        'input_ids': torch.zeros(2, 3, dtype=torch.long),
        'attention_mask': torch.ones(2, 3),
        'embodiment_id': torch.ones(2, dtype=torch.long),
        'state': torch.zeros(2, 3), 'state_mask': torch.ones(2, 3),
        'action_hz': torch.full((2,), 30),
        'actions': torch.full((2, 4, 3), 2.0),
        'actions_mask': torch.ones(2, 4, 3, dtype=torch.bool),
    }


class DiagnosticTests(unittest.TestCase):
    def test_noise_pairing_metrics_and_rng_restoration(self):
        backend = FakeBackend().eval()
        state = torch.get_rng_state().clone()
        metrics = diagnose_train_batch(backend, fake_batch(), noise_repeats=2, prefix_steps=2)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for start in (0, 3):
            self.assertTrue(torch.equal(backend.flow.noises[start], backend.flow.noises[start + 1]))
            self.assertTrue(torch.equal(backend.flow.noises[start], backend.flow.noises[start + 2]))
        self.assertFalse(torch.equal(backend.flow.noises[0], backend.flow.noises[3]))
        self.assertTrue(torch.equal(metrics['latent_mse'], torch.ones(2)))
        self.assertTrue(torch.equal(metrics['future_mse/predicted'], torch.full((2,), 4.0)))
        self.assertTrue(torch.equal(metrics['future_mse/teacher_latent'], torch.ones(2)))
        self.assertTrue(torch.equal(metrics['future_mse/gt_future'], torch.zeros(2)))
        self.assertTrue(torch.equal(metrics['action_change_mse/predicted'], torch.zeros(2)))
        self.assertTrue((metrics['action_mse/teacher_latent/all'] < metrics['action_mse/predicted/all']).all())
        self.assertTrue((metrics['action_mse/gt_future/all'] < metrics['action_mse/teacher_latent/all']).all())

    def test_masked_dimensions_do_not_contribute(self):
        prediction = torch.tensor([[[1.0, 100.0], [3.0, 100.0]]])
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[[True, False], [True, False]]])
        self.assertAlmostEqual(masked_error_per_sample(prediction, target, mask).item(), 5.0)
        with self.assertRaises(ValueError):
            masked_error_per_sample(prediction, target, mask & False)

    def test_requires_eval_complete_future_and_positive_repeats(self):
        backend = FakeBackend()
        with self.assertRaises(ValueError):
            diagnose_train_batch(backend, fake_batch())
        backend.eval()
        with self.assertRaises(ValueError):
            diagnose_train_batch(backend, fake_batch(), noise_repeats=0)
        batch = fake_batch()
        batch['primary_video'] = batch['primary_video'][:, :1]
        with self.assertRaises(ValueError):
            diagnose_train_batch(backend, batch)
        backend.model_cfg.flow_future_feature = 'current'
        with self.assertRaises(ValueError):
            diagnose_train_batch(backend, fake_batch())

    def test_actual_backend_baseline_matches_normal_inference(self):
        for mode in ('vlm_queries', 'external_qformer'):
            with self.subTest(mode=mode):
                torch.manual_seed(4)
                backend = build_tiny_backend(mode).eval()
                batch = collate(backend, [raw_sample(), raw_sample()])
                result = diagnose_train_batch(backend, batch, noise_seed=17, noise_repeats=1)
                infer = {key: value for key, value in batch.items() if key not in ('actions', 'actions_mask', 'primary_video')}
                infer['primary_image'] = batch['primary_video'][:, 0]
                torch.manual_seed(17)
                prediction = backend.predict_action(batch=infer, return_padded=True, guidance_scale=1.0)
                expected = masked_error_per_sample(prediction, batch['actions'], batch['actions_mask'])
                torch.testing.assert_close(result['action_mse/predicted/all'], expected, atol=1e-6, rtol=1e-6)
                changed = dict(batch)
                changed['primary_video'] = batch['primary_video'].clone()
                changed['primary_video'][:, -1] += 2
                second = diagnose_train_batch(backend, changed, noise_seed=17, noise_repeats=1)
                torch.testing.assert_close(result['action_mse/predicted/all'], second['action_mse/predicted/all'])
                self.assertFalse(torch.equal(result['latent_mse'], second['latent_mse']))


class WindowAndReportTests(unittest.TestCase):
    def mixture(self):
        dataset = SimpleNamespace(
            dataset_name='bimanual_photograph',
            transforms=SimpleNamespace(eval=lambda: None),
            delta_indices={'video.front': np.array([0, 3]), 'action.proprio': np.arange(4)},
            modality_keys={'video': ['video.front']}, action_hz=30.0,
            trajectory_ids=np.array([99, 100]), trajectory_lengths=np.array([8, 3]),
        )
        return SimpleNamespace(datasets=[dataset])

    def test_windows_exclude_padded_futures_and_keep_split(self):
        dataset = FixedEpisodeWindows(self.mixture(), windows_per_episode=64)
        self.assertEqual([w['start_frame'] for w in dataset.windows], [0, 1, 2, 3, 4])
        self.assertEqual(dataset.windows[-1]['future_frame'], 7)
        self.assertEqual(len(dataset.skipped_episodes), 1)
        with self.assertRaises(ValueError):
            FixedEpisodeWindows(self.mixture(), episode_ids=[1])

    def test_uniform_subsampling(self):
        dataset = FixedEpisodeWindows(self.mixture(), windows_per_episode=3)
        self.assertEqual([w['start_frame'] for w in dataset.windows], [0, 2, 4])

    def test_checkpoint_directory_and_paired_episode_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / 'final_model').mkdir()
            path = run / 'final_model/pytorch_model.pt'
            path.touch()
            self.assertEqual(resolve_checkpoint(run), path)
        rows = []
        for episode, baseline, teacher, gt in [(99, 2.0, 1.0, 0.5), (100, 4.0, 2.0, 1.0)]:
            rows.append({
                'dataset': 'task', 'episode_id': episode, 'start_frame': 0, 'future_frame': 3,
                'action_mse/predicted/all': baseline,
                'action_mse/teacher_latent/all': teacher,
                'action_mse/gt_future/all': gt,
            })
        report = summarize(rows)
        self.assertEqual(report['paired_improvement']['teacher_latent']['episode_macro_mse_reduction'], 1.5)
        self.assertEqual(report['paired_improvement']['teacher_latent']['episodes_improved'], 2)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
