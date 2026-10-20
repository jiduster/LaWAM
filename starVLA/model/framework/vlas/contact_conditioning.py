"""Current oracle contact geometry; independent of VLM queries and future labels."""

import torch
from torch import nn


class ContactEncoder(nn.Module):
    """Encode [target xyz + outward normal, eight fingertip xyz + zeros].

    Inputs are [B,9,6] in MuJoCo world coordinates (metres). Finger order is
    left FF/MF/RF/TH, right FF/MF/RF/TH. Output order is target, tips, relations.
    """

    def __init__(self, dim: int, position_scale: float = 0.1):
        super().__init__()
        if position_scale <= 0:
            raise ValueError("contact_position_scale must be positive")
        self.position_scale = float(position_scale)
        self.target = nn.Sequential(nn.Linear(6, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.tip = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.relation = nn.Sequential(nn.Linear(5, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.identity = nn.Parameter(torch.randn(1, 17, dim) * 0.02)

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        if geometry.ndim != 3 or geometry.shape[1:] != (9, 6):
            raise ValueError(f"contact_geometry must be [B,9,6], got {tuple(geometry.shape)}")
        if not torch.isfinite(geometry).all():
            raise ValueError("contact_geometry contains non-finite values")
        # Retain metric precision before casting into the action expert's dtype.
        g = geometry.float()
        p, normal = g[:, :1, :3], g[:, :1, 3:]
        if not torch.allclose(normal.norm(dim=-1), torch.ones_like(normal[..., 0]), atol=1e-3):
            raise ValueError("Contact target normal must be a unit vector")
        if torch.any(g[:, 1:, 3:] != 0):
            raise ValueError("Fingertip normal slots are reserved and must be zero")
        tips = g[:, 1:, :3]
        delta = (p - tips) / self.position_scale
        relation = torch.cat((delta, delta.norm(dim=-1, keepdim=True),
                              (delta * normal).sum(dim=-1, keepdim=True)), dim=-1)
        dtype = self.identity.dtype
        target = torch.cat((p / self.position_scale, normal), dim=-1)
        return torch.cat((self.target(target.to(dtype)),
                          self.tip((tips / self.position_scale).to(dtype)),
                          self.relation(relation.to(dtype))), dim=1) + self.identity


class ContactCrossAttention(nn.Module):
    """Zero-output residual initialization preserves the pretrained policy."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, hidden: torch.Tensor, tokens: torch.Tensor,
                valid: torch.Tensor | None = None) -> torch.Tensor:
        context = self.context_norm(tokens)
        residual = self.attention(self.norm(hidden), context, context, need_weights=False)[0]
        if valid is not None:
            residual = residual * valid.unsqueeze(-1).to(residual.dtype)
        return hidden + residual
