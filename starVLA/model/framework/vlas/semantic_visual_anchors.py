"""Task-agnostic DINO patch clustering with BC-trained task/state conditioning."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_visual_anchor_config(cfg):
    if not cfg.visual_anchor_conditioning:
        return
    for name in ("visual_anchor_num_tokens", "visual_anchor_hidden_dim", "visual_anchor_kmeans_iters"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not math.isfinite(cfg.visual_anchor_spatial_weight) or cfg.visual_anchor_spatial_weight <= 0:
        raise ValueError("visual_anchor_spatial_weight must be positive and finite.")
    interval = cfg.flow_cfg.anchor_attention_every_n_blocks
    if not 1 <= interval <= cfg.flow_cfg.num_layers:
        raise ValueError("Anchor attention interval must be within the number of action-expert blocks.")


class SemanticVisualAnchors(nn.Module):
    """Select current-image regions; encode descriptors, UV, task, and proprioception.

    UV is in [-1, 1] in the primary DINO image, not a metric 3D position.
    Candidate identities are unordered and recomputed at every observation.
    Only current features enter proposals; no future images or labels are used.
    """

    def __init__(self, *, vision_dim, vlm_dim, state_dim, token_dim, grid_height,
                 grid_width, num_tokens=16, hidden_dim=256, kmeans_iters=4, spatial_weight=0.5):
        super().__init__()
        self.grid_height, self.grid_width = int(grid_height), int(grid_width)
        if min(self.grid_height, self.grid_width) <= 0:
            raise ValueError("Visual anchors require the LAM encoder's spatial grid dimensions.")
        if not 1 <= num_tokens <= self.grid_height * self.grid_width:
            raise ValueError("visual_anchor_num_tokens exceeds the number of DINO patches.")
        self.num_tokens = int(num_tokens)
        self.kmeans_iters = int(kmeans_iters)
        self.spatial_weight = float(spatial_weight)
        self.vision_dim, self.state_dim = int(vision_dim), int(state_dim)
        y, x = torch.meshgrid(
            (torch.arange(self.grid_height).float() + 0.5) * (2 / self.grid_height) - 1,
            (torch.arange(self.grid_width).float() + 0.5) * (2 / self.grid_width) - 1,
            indexing="ij",
        )
        self.register_buffer("patch_uv", torch.stack((x, y), dim=-1).flatten(0, 1), persistent=False)
        self.visual_encoder = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, hidden_dim), nn.GELU())
        self.uv_encoder = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU())
        self.task_encoder = nn.Sequential(nn.LayerNorm(vlm_dim), nn.Linear(vlm_dim, hidden_dim), nn.GELU())
        self.state_encoder = nn.Sequential(nn.Linear(state_dim * 2, hidden_dim), nn.GELU())
        self.relation_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, token_dim),
        )
        self.importance_head = nn.Linear(token_dim, 1)
        nn.init.zeros_(self.importance_head.weight)
        nn.init.zeros_(self.importance_head.bias)

    @torch.no_grad()
    def propose(self, features):
        if features.ndim != 3 or features.shape[1:] != (self.patch_uv.shape[0], self.vision_dim):
            raise ValueError("Visual anchors expect current DINO tokens [B,grid_height*grid_width,vision_dim].")
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite current DINO features cannot define visual anchors.")
        batch, count, _ = features.shape
        uv = self.patch_uv.float().unsqueeze(0).expand(batch, -1, -1)
        descriptors = F.normalize(features.detach().float(), dim=-1)
        points = torch.cat((descriptors, uv * self.spatial_weight), dim=-1)
        batch_ids = torch.arange(batch, device=features.device)
        # Deterministic farthest-point seeds avoid stochastic train/inference proposals.
        distances = (points - points.mean(dim=1, keepdim=True)).square().sum(dim=-1)
        selected = torch.zeros(batch, count, dtype=torch.bool, device=features.device)
        seed_indices = []
        minimum = torch.full_like(distances, float("inf"))
        for _ in range(self.num_tokens):
            index = distances.masked_fill(selected, -1).argmax(dim=1)
            seed_indices.append(index)
            selected[batch_ids, index] = True
            delta = (points - points[batch_ids, index].unsqueeze(1)).square().sum(dim=-1)
            minimum = torch.minimum(minimum, delta)
            distances = minimum
        seeds = torch.stack(seed_indices, dim=1)
        centers = points.gather(1, seeds.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
        for _ in range(self.kmeans_iters):
            distances = (points.square().sum(-1, keepdim=True)
                         + centers.square().sum(-1).unsqueeze(1)
                         - 2 * torch.bmm(points, centers.transpose(1, 2)))
            labels = distances.argmin(dim=-1)
            membership = F.one_hot(labels, self.num_tokens).transpose(1, 2).float()
            sizes = membership.sum(dim=-1, keepdim=True)
            fallback = F.one_hot(seeds, count).float()
            membership = torch.where(sizes > 0, membership, fallback)
            weights = membership / membership.sum(dim=-1, keepdim=True)
            centers = torch.bmm(weights, points)
        # Pick a real patch nearest the centroid; pool each region's descriptor.
        patch_distances = (points.square().sum(-1).unsqueeze(1)
                           + centers.square().sum(-1, keepdim=True)
                           - 2 * torch.bmm(centers, points.transpose(1, 2)))
        representatives = patch_distances.masked_fill(~membership.bool(), float("inf")).argmin(dim=-1)
        anchor_uv = uv.gather(1, representatives.unsqueeze(-1).expand(-1, -1, 2))
        return weights, anchor_uv, representatives

    def forward(self, *, current_features, semantic_features, semantic_mask, state, state_mask):
        weights, uv, indices = self.propose(current_features)
        dtype = self.visual_encoder[1].weight.dtype
        features = torch.bmm(weights.to(dtype), current_features.to(dtype))
        valid = semantic_mask.bool().unsqueeze(-1)
        semantic = semantic_features.to(dtype).masked_fill(~valid, 0).sum(dim=1)
        semantic = semantic / valid.sum(dim=1).clamp_min(1).to(dtype)
        valid_state = state_mask.bool()
        state_input = torch.cat((state.to(dtype).masked_fill(~valid_state, 0), valid_state.to(dtype)), dim=-1)
        task_context = self.task_encoder(semantic).unsqueeze(1).expand(-1, self.num_tokens, -1)
        hand_context = self.state_encoder(state_input).unsqueeze(1).expand(-1, self.num_tokens, -1)
        tokens = self.relation_encoder(torch.cat((
            self.visual_encoder(features), self.uv_encoder(uv.to(dtype)), task_context, hand_context,
        ), dim=-1))
        importance = self.importance_head(tokens).squeeze(-1).softmax(dim=1)
        # A weighted context changes feature direction and survives the action
        # attention's LayerNorm; scalar importance alone would be normalized away.
        tokens = tokens + (tokens * importance.unsqueeze(-1)).sum(dim=1, keepdim=True)
        return tokens, {
            "visual_anchor_uv": uv, "visual_anchor_patch_indices": indices,
            "visual_anchor_importance": importance,
        }
