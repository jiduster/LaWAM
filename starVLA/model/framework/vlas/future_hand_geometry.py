"""Predict explicit hand geometry and encode it as action-conditioning tokens."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_geometry_config(cfg) -> None:
    if not cfg.geometry_conditioning:
        if cfg.geometry_loss_weight != 0:
            raise ValueError("geometry_loss_weight requires geometry_conditioning=true.")
        return
    indices = tuple(cfg.geometry_frame_indices)
    if not indices or any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
        raise ValueError("geometry_frame_indices must contain integer action-frame offsets.")
    if tuple(sorted(set(indices))) != indices or indices[0] < 1 or indices[-1] >= cfg.action_horizon:
        raise ValueError("geometry_frame_indices must be unique, increasing future offsets within action_horizon.")
    if cfg.past_action_window_size != 0 or not cfg.future_prediction:
        raise ValueError("Geometry conditioning requires past_action_window_size=0 and future_prediction=true.")
    if str(cfg.flow_future_feature).lower() != "predicted":
        raise ValueError("Geometry conditioning requires flow_future_feature=predicted.")
    if not math.isfinite(cfg.geometry_loss_weight) or cfg.geometry_loss_weight <= 0:
        raise ValueError("geometry_loss_weight must be positive when geometry conditioning is enabled.")
    if not math.isfinite(cfg.geometry_target_scale) or cfg.geometry_target_scale <= 0:
        raise ValueError("geometry_target_scale must be positive and finite.")
    if len(cfg.geometry_target_center) != 3 or not all(math.isfinite(v) for v in cfg.geometry_target_center):
        raise ValueError("geometry_target_center must contain three finite coordinates in meters.")
    if cfg.geometry_hidden_dim <= 0 or cfg.flow_cfg.geometry_attention_every_n_blocks <= 0:
        raise ValueError("Geometry hidden dimension and attention interval must be positive.")
    if cfg.flow_cfg.geometry_attention_every_n_blocks > cfg.flow_cfg.num_layers:
        raise ValueError("Geometry attention interval exceeds the number of action-expert blocks.")


class FutureHandGeometry(nn.Module):
    """Eight Allegro tips in fixed-camera coordinates; left then right, ff/mf/rf/th."""

    def __init__(self, *, vision_dim, state_dim, hidden_dim, token_dim, frame_indices, center, scale, condition_actions=True):
        super().__init__()
        self.frame_indices = tuple(frame_indices)
        self.scale = float(scale)
        self.condition_actions = condition_actions
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32), persistent=False)
        self.position_head = nn.Sequential(
            nn.Linear(vision_dim + state_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, len(self.frame_indices) * 8 * 3),
        )
        if not condition_actions:
            return
        self.coordinate_encoder = nn.Sequential(nn.Linear(3, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))
        self.time_encoder = nn.Sequential(nn.Linear(1, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))
        self.hand_embedding = nn.Embedding(2, token_dim)
        self.finger_embedding = nn.Embedding(4, token_dim)
        nn.init.normal_(self.hand_embedding.weight, std=0.02)
        nn.init.normal_(self.finger_embedding.weight, std=0.02)

    def forward(self, future_features, state, state_mask, action_hz):
        dtype = self.position_head[0].weight.dtype
        masked_state = state.to(dtype).masked_fill(~state_mask.bool(), 0)
        context = torch.cat((future_features.to(dtype).mean(dim=1), masked_state), dim=-1)
        positions = self.position_head(context).view(context.shape[0], len(self.frame_indices), 8, 3)
        return positions, self.encode_tokens(positions, action_hz) if self.condition_actions else None

    def encode_tokens(self, normalized_positions, action_hz):
        if normalized_positions.shape[1:] != (len(self.frame_indices), 8, 3):
            raise ValueError("Expected normalized geometry [B,num_times,8,3].")
        if action_hz.shape != (normalized_positions.shape[0],) or not torch.all(torch.isfinite(action_hz) & (action_hz > 0)):
            raise ValueError("Geometry action_hz must contain one positive finite rate per sample.")
        device, dtype = normalized_positions.device, normalized_positions.dtype
        offsets = normalized_positions.new_tensor(self.frame_indices)
        seconds = offsets.unsqueeze(0) / action_hz.to(device=device, dtype=dtype).unsqueeze(1)
        identities = torch.arange(8, device=device)
        tokens = (
            self.coordinate_encoder(normalized_positions)
            + self.time_encoder(seconds.unsqueeze(-1)).unsqueeze(2)
            + self.hand_embedding(identities // 4)[None, None]
            + self.finger_embedding(identities % 4)[None, None]
        )
        return tokens.flatten(1, 2)

    def positions_in_meters(self, normalized_positions):
        return normalized_positions.float() * self.scale + self.center.float()

    def supervised_loss(self, normalized_positions, targets, valid):
        if targets.shape != normalized_positions.shape or valid.shape != targets.shape[:-1]:
            raise ValueError("Geometry targets/mask must match predicted [B,num_times,8,3] geometry.")
        valid = valid.bool() & torch.isfinite(targets).all(dim=-1)
        safe_targets = torch.where(valid.unsqueeze(-1), targets.float(), self.center.float())
        normalized_targets = (safe_targets - self.center.float()) / self.scale
        error = F.smooth_l1_loss(normalized_positions.float(), normalized_targets, reduction="none")
        denominator = valid.sum().clamp_min(1) * 3
        loss = error.masked_fill(~valid.unsqueeze(-1), 0).sum() / denominator
        squared = (self.positions_in_meters(normalized_positions) - safe_targets).square()
        rmse_mm = (squared.masked_fill(~valid.unsqueeze(-1), 0).sum() / denominator).sqrt() * 1000
        return loss, rmse_mm.detach(), valid.float().mean().detach()
