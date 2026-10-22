from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.latent_world import (
    LatentWorldPolicyInferExample,
    LatentWorldPolicyTrainBatch,
    build_policy_components,
)
from starVLA.model.framework.latent_world.runtime.freeze_policy import (
    apply_policy_freeze,
    parse_policy_freeze_config,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("LaWAM")
@FRAMEWORK_REGISTRY.register("lawam")
class LaWAMFramework(baseframework):
    def __init__(self, config: Any = None, **kwargs) -> None:
        super().__init__()
        del kwargs

        self.config = config
        components = build_policy_components(config)

        self.policy_cfg = components.policy_cfg
        self.policy_backend = components.policy_backend
        self.policy_vlm_adapter = components.policy_vlm_adapter
        self.policy_infer_batch_builder = components.infer_batch_builder
        self.policy_runner = components.runner
        self.policy_action_head = self.policy_backend.flow

        self.processor = self.policy_backend.processor

    def apply_training_freeze_policy(self, freeze_cfg) -> None:
        freeze_policy = parse_policy_freeze_config(freeze_cfg)
        apply_policy_freeze(self.policy_backend, freeze_policy)
        if self.policy_cfg.action_expert_only:
            self.policy_backend.configure_action_expert_only()
        self.policy_backend.sync_training_modes(mode=self.training)

    def forward(self, batch: LatentWorldPolicyTrainBatch, **kwargs) -> Dict[str, torch.Tensor]:
        del kwargs
        return self.policy_runner.train_step(batch)

    @torch.inference_mode()
    def predict_action(self, examples: Sequence[LatentWorldPolicyInferExample], **kwargs) -> Dict[str, Any]:
        return self.policy_runner.infer_step(examples, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        return self.policy_backend.save_pretrained(save_directory, **kwargs)
