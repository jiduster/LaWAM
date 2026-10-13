from __future__ import annotations

import torch
import torch.nn as nn


class LatentActionHead(nn.Module):
    """Predict one LAM latent from VLM tokens using an external learned query.

    Parameter names and shapes match the pretrained VLMToLAMQFormer so the
    head can reuse its query, attention, FFN, and normalization weights.
    ``attention_mask`` uses True/1 for valid context tokens, as in the VLM.
    """

    def __init__(
        self,
        *,
        vlm_hidden_dim: int,
        lam_code_dim: int,
        num_layers: int = 1,
        num_heads: int = 8,
        ffn_expansion_factor: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, 1, int(lam_code_dim)) * 0.02)
        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=int(lam_code_dim),
                    kdim=int(vlm_hidden_dim),
                    vdim=int(vlm_hidden_dim),
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm_qs = nn.ModuleList([nn.LayerNorm(int(lam_code_dim)) for _ in range(int(num_layers))])
        self.norm_kvs = nn.ModuleList([nn.LayerNorm(int(vlm_hidden_dim)) for _ in range(int(num_layers))])
        hidden_dim = int(int(lam_code_dim) * float(ffn_expansion_factor))
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(int(lam_code_dim)),
                    nn.Linear(int(lam_code_dim), hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden_dim, int(lam_code_dim)),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(int(lam_code_dim))

    def forward(
        self,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context.ndim != 3 or context.shape[1] == 0:
            raise ValueError(f"Latent action context must be nonempty [B,L,D], got {tuple(context.shape)}.")
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != context.shape[:2]:
                raise ValueError(
                    f"Latent action attention_mask {tuple(attention_mask.shape)} "
                    f"does not match context {tuple(context.shape[:2])}."
                )
            valid = attention_mask.to(device=context.device, dtype=torch.bool)
            if not valid.any(dim=1).all():
                raise ValueError("Every latent action context must contain at least one valid token.")
            key_padding_mask = ~valid

        bsz = int(context.shape[0])
        queries = self.query.expand(bsz, -1, -1).to(device=context.device, dtype=context.dtype)
        for norm_q, norm_kv, xattn, ffn in zip(self.norm_qs, self.norm_kvs, self.cross_attns, self.ffns):
            q = norm_q(queries)
            kv = norm_kv(context)
            attn_out, _ = xattn(q, kv, kv, key_padding_mask=key_padding_mask)
            queries = queries + attn_out
            queries = queries + ffn(queries)
        return self.final_norm(queries)
