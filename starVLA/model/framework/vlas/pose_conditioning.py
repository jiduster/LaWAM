"""Independent current-pose encoders and residual attention branches."""

import torch
from torch import nn


def pose_conditioning_enabled(policy_cfg):
    return (bool(getattr(policy_cfg, "plan_pose_conditioning", False))
            or bool(getattr(policy_cfg.flow_cfg, "pose_conditioning", False)))


def validate_pose_tensor(pose: torch.Tensor) -> None:
    if pose.ndim != 2 or pose.shape[1] != 12:
        raise ValueError(f"object_pose_condition must be [B,12], got {tuple(pose.shape)}")
    if not torch.isfinite(pose).all():
        raise ValueError("object_pose_condition contains non-finite values")
    if not torch.allclose(pose[:, 9:12].float().norm(dim=-1),
                          torch.ones(pose.shape[0], device=pose.device), atol=1e-3, rtol=0):
        raise ValueError("object_pose_condition goal direction must be unit length")


class PoseEncoder(nn.Module):
    def __init__(self, dim, hidden_dim, position_mean, position_std):
        super().__init__()
        mean = torch.as_tensor(position_mean, dtype=torch.float32)
        std = torch.as_tensor(position_std, dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Pose position normalization requires finite training mean[3] and positive std[3]")
        if hidden_dim <= 0:
            raise ValueError("pose_encoder_hidden_dim must be positive")
        self.register_buffer("position_mean", mean)
        self.register_buffer("position_std", std)
        self.groups = nn.ModuleList([
            nn.Sequential(nn.Linear(n, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))
            for n in (3, 6, 3)
        ])
        self.type_embedding = nn.Parameter(torch.randn(1, 3, dim) * 0.02)

    def forward(self, pose):
        validate_pose_tensor(pose)
        pose = pose.to(device=self.position_mean.device, dtype=torch.float32)
        groups = ((pose[:, :3] - self.position_mean.float()) / self.position_std.float(),
                  pose[:, 3:9], pose[:, 9:12])
        return torch.stack([mlp(x.to(self.type_embedding.dtype))
                            for mlp, x in zip(self.groups, groups)], dim=1) + self.type_embedding


class PoseCrossAttention(nn.Module):
    """Independent parameters; zero output preserves the pretrained action expert."""

    def __init__(self, dim, heads):
        super().__init__()
        if heads <= 0 or dim % heads:
            raise ValueError("pose_attention_heads must divide DiT hidden_dim")
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, hidden, tokens, valid=None):
        context = self.context_norm(tokens)
        residual = self.attention(self.norm(hidden), context, context, need_weights=False)[0]
        if valid is not None:
            residual = residual * valid.unsqueeze(-1).to(residual.dtype)
        return hidden + residual


class PlanningPoseAdapter(nn.Module):
    """Fuse pose into a private planning feature branch with a small attention width."""

    def __init__(self, feature_dim, token_dim, heads):
        super().__init__()
        if token_dim <= 0 or heads <= 0 or token_dim % heads:
            raise ValueError("plan_pose_token_dim must be positive and divisible by plan_pose_attention_heads")
        self.norm = nn.LayerNorm(feature_dim)
        self.context_norm = nn.LayerNorm(token_dim)
        self.query_projection = nn.Linear(feature_dim, token_dim)
        self.attention = nn.MultiheadAttention(token_dim, heads, dropout=0.0, batch_first=True)
        self.output_projection = nn.Linear(token_dim, feature_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, features, pose_tokens, valid=None):
        if pose_tokens is None:
            raise KeyError("Planning pose adapter requires current object_pose_condition")
        if features.ndim != 3 or pose_tokens.ndim != 3 or features.shape[0] != pose_tokens.shape[0]:
            raise ValueError("Planning features and pose tokens must have matching [B,L,D] batches")
        query = self.query_projection(self.norm(features))
        context = self.context_norm(pose_tokens.to(device=features.device, dtype=query.dtype))
        residual = self.attention(query, context, context, need_weights=False)[0]
        residual = self.output_projection(residual)
        if valid is not None:
            residual = residual * valid.unsqueeze(-1).to(residual.dtype)
        return features + residual
