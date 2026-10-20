"""LingBot-style depth queries for the LaWAM VLM.

The 256 learnable vectors are mean-pooled into 8 tokens and copied once per
camera. Those tokens are written onto prompt placeholders. Their VLM hidden
states stay inside ``h_vlm``, so the action flow sees them. The 256 x 1024
resampler output is compared with frozen MoGe-2 + LingBot-Depth features and
is not written back into the VLM or the flow.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

DEPTH_VIEW_NAMES = ("primary_view", "wrist_left", "wrist_right")
DEPTH_TARGET_TOKENS = 256
DEPTH_TARGET_DIM = 1024


DEFAULT_DEPTH_TEACHER_ROOT = (
    "/data/home/zyh/lingbot-vla-v2/lingbotvla/models/vla/vision_models"
)


def _feed_forward(dim: int, mult: int) -> nn.Sequential:
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def _split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    batch, length, width = x.shape
    x = x.view(batch, length, heads, width // heads)
    return x.permute(0, 2, 1, 3)


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = int(dim_head)
        self.heads = int(heads)
        inner_dim = int(dim_head * heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm1(x)
        latents = self.norm2(latents)
        batch, length, _ = latents.shape
        query = _split_heads(self.to_q(latents), self.heads)
        key, value = self.to_kv(torch.cat((x, latents), dim=-2)).chunk(2, dim=-1)
        key = _split_heads(key, self.heads)
        value = _split_heads(value, self.heads)
        scale = 1.0 / math.sqrt(math.sqrt(self.dim_head))
        weight = (query * scale) @ (key * scale).transpose(-2, -1)
        if context_mask is not None:
            latent_valid = torch.ones(
                batch, length, dtype=torch.bool, device=x.device
            )
            key_valid = torch.cat([context_mask.to(dtype=torch.bool, device=x.device), latent_valid], dim=1)
            weight = weight.masked_fill(~key_valid[:, None, None, :], torch.finfo(weight.dtype).min)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ value
        out = out.permute(0, 2, 1, 3).reshape(batch, length, -1)
        return self.to_out(out)


class TaskTokenResampler(nn.Module):
    def __init__(
        self,
        *,
        dim_in: int,
        dim_mid: int,
        dim_head: int,
        dim_out: int,
        num_layers: int,
        num_heads: int,
        ff_mult: int,
    ) -> None:
        super().__init__()
        self.proj_in1 = nn.Linear(dim_in, dim_mid)
        self.proj_in2 = nn.Linear(dim_in, dim_mid)
        self.proj_out = nn.Linear(dim_mid, dim_out)
        self.norm_out = nn.LayerNorm(dim_out)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim_mid, dim_head=dim_head, heads=num_heads),
                        _feed_forward(dim_mid, ff_mult),
                    ]
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        queries: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = self.proj_in1(queries)
        x = self.proj_in2(x)
        for attn, ff in self.layers:
            queries = attn(x, queries, context_mask=context_mask) + queries
            queries = ff(queries) + queries
        return self.norm_out(self.proj_out(queries))


class TaskTokenDepthHead(nn.Module):
    """256 VLM-width queries resampled to LingBot-Depth's 1024-d features."""

    def __init__(
        self,
        *,
        vlm_dim: int,
        teacher_dim: int,
        num_backbone_tokens: int,
        num_task_tokens: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dim_head: int = 32,
        ff_mult: int = 1,
    ) -> None:
        super().__init__()
        if int(num_task_tokens) <= 0 or int(num_backbone_tokens) % int(num_task_tokens) != 0:
            raise ValueError(
                "depth backbone tokens must be divisible by the per-view task tokens, "
                f"got backbone={num_backbone_tokens}, task={num_task_tokens}."
            )
        self.num_backbone_tokens = int(num_backbone_tokens)
        self.num_task_tokens = int(num_task_tokens)
        self.align_embs = nn.Parameter(torch.randn(self.num_backbone_tokens, int(vlm_dim)) * 0.02)
        self.projector = TaskTokenResampler(
            dim_in=int(vlm_dim),
            dim_mid=int(vlm_dim),
            dim_head=int(dim_head),
            dim_out=int(teacher_dim),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            ff_mult=int(ff_mult),
        )

    def pooled_task_tokens(self) -> torch.Tensor:
        grouped = self.align_embs.view(
            self.num_task_tokens,
            self.num_backbone_tokens // self.num_task_tokens,
            self.align_embs.shape[-1],
        )
        return grouped.mean(dim=1)

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = self.align_embs.to(dtype=context.dtype, device=context.device)
        queries = queries.unsqueeze(0).expand(context.shape[0], -1, -1)
        return self.projector(context, queries, context_mask=context_mask)


def image_token_spans(
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    image_grid_thw: torch.Tensor,
    merge_size: int,
    views_per_sample: list[int],
) -> list[list[tuple[int, int]]]:
    if image_grid_thw.ndim != 2 or int(image_grid_thw.shape[-1]) != 3:
        raise ValueError(f"image_grid_thw must have shape [N, 3], got {tuple(image_grid_thw.shape)}.")
    counts = (image_grid_thw.long().prod(dim=-1) // int(merge_size) ** 2).tolist()
    if sum(views_per_sample) != len(counts):
        raise ValueError(
            "Image-grid count does not match depth-query views: "
            f"grids={len(counts)}, views={sum(views_per_sample)}."
        )
    spans: list[list[tuple[int, int]]] = []
    cursor = 0
    for batch_index, view_count in enumerate(views_per_sample):
        positions = torch.nonzero(input_ids[batch_index] == int(image_token_id), as_tuple=False).flatten()
        sample_spans: list[tuple[int, int]] = []
        offset = 0
        for count in counts[cursor : cursor + view_count]:
            count = int(count)
            chunk = positions[offset : offset + count]
            if int(chunk.numel()) != count:
                raise ValueError(
                    f"Sample {batch_index} ran out of image tokens while splitting view grids."
                )
            start = int(chunk[0].item())
            end = int(chunk[-1].item()) + 1
            if end - start != count or not torch.equal(
                chunk, torch.arange(start, end, device=chunk.device)
            ):
                raise ValueError(
                    f"Sample {batch_index} image tokens for one view are not contiguous."
                )
            sample_spans.append((start, end))
            offset += count
        if offset != int(positions.numel()):
            raise ValueError(
                f"Sample {batch_index} has {int(positions.numel())} image tokens, "
                f"but the depth views account for {offset}."
            )
        spans.append(sample_spans)
        cursor += view_count
    return spans


def depth_placeholder_spans(
    depth_mask: torch.Tensor,
    *,
    tokens_per_view: int,
) -> list[list[tuple[int, int]]]:
    spans: list[list[tuple[int, int]]] = []
    tokens_per_view = int(tokens_per_view)
    for batch_index in range(int(depth_mask.shape[0])):
        positions = torch.nonzero(depth_mask[batch_index], as_tuple=False).flatten()
        if int(positions.numel()) % tokens_per_view != 0:
            raise ValueError(
                f"Sample {batch_index} has {int(positions.numel())} depth placeholders, "
                f"which is not a multiple of {tokens_per_view}."
            )
        sample_spans: list[tuple[int, int]] = []
        view_count = int(positions.numel()) // tokens_per_view
        for view_index in range(view_count):
            chunk = positions[view_index * tokens_per_view : (view_index + 1) * tokens_per_view]
            start = int(chunk[0].item())
            end = int(chunk[-1].item()) + 1
            if end - start != tokens_per_view or not torch.equal(
                chunk, torch.arange(start, end, device=chunk.device)
            ):
                raise ValueError(
                    f"Sample {batch_index} view {view_index} depth placeholders are not contiguous."
                )
            sample_spans.append((start, end))
        spans.append(sample_spans)
    return spans


def build_depth_query_attention_mask(
    attention_mask: torch.Tensor,
    image_spans: list[list[tuple[int, int]]],
    depth_spans: list[list[tuple[int, int]]],
) -> torch.Tensor:
    """Bool mask, True means the query may attend to the key.

    Non-query tokens keep a causal padding mask, except they cannot see depth
    queries. Query group i sees view i's image tokens and its own group.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be [B, S], got {tuple(attention_mask.shape)}.")
    batch, seq_len = attention_mask.shape
    if len(image_spans) != batch or len(depth_spans) != batch:
        raise ValueError("Span lists must contain one entry per batch row.")
    device = attention_mask.device
    valid = attention_mask.to(device=device, dtype=torch.bool)
    positions = torch.arange(seq_len, device=device)
    allow = (positions.view(seq_len, 1) >= positions.view(1, seq_len)) & valid[:, :, None] & valid[:, None, :]
    allow = allow.clone()
    for batch_index in range(batch):
        if len(image_spans[batch_index]) != len(depth_spans[batch_index]):
            raise ValueError(
                f"Sample {batch_index} has {len(image_spans[batch_index])} image views and "
                f"{len(depth_spans[batch_index])} depth-query groups."
            )
        query_positions: list[int] = []
        for start, end in depth_spans[batch_index]:
            query_positions.extend(range(start, end))
        if query_positions:
            query_index = torch.tensor(query_positions, dtype=torch.long, device=device)
            allow[batch_index, :, query_index] = False
            allow[batch_index, query_index, :] = False
        for (image_start, image_end), (query_start, query_end) in zip(
            image_spans[batch_index], depth_spans[batch_index]
        ):
            allow[batch_index, query_start:query_end, image_start:image_end] = True
            allow[batch_index, query_start:query_end, query_start:query_end] = True
        allow[batch_index, ~valid[batch_index], :] = False
        allow[batch_index, :, ~valid[batch_index]] = False
    return allow.unsqueeze(1)


def gather_depth_contexts(
    hidden: torch.Tensor,
    image_spans: list[list[tuple[int, int]]],
    depth_spans: list[list[tuple[int, int]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    contexts: list[torch.Tensor] = []
    for batch_index, (images, queries) in enumerate(zip(image_spans, depth_spans)):
        for (image_start, image_end), (query_start, query_end) in zip(images, queries):
            contexts.append(
                torch.cat(
                    [
                        hidden[batch_index, image_start:image_end],
                        hidden[batch_index, query_start:query_end],
                    ],
                    dim=0,
                )
            )
    if not contexts:
        raise ValueError("Depth context gathering found no views.")
    width = int(hidden.shape[-1])
    max_len = max(int(item.shape[0]) for item in contexts)
    packed = hidden.new_zeros(len(contexts), max_len, width)
    mask = torch.zeros(len(contexts), max_len, dtype=torch.bool, device=hidden.device)
    for index, item in enumerate(contexts):
        packed[index, : item.shape[0]] = item
        mask[index, : item.shape[0]] = True
    return packed, mask


class FrozenDepthTeacher:
    """Online MoGe-2 then LingBot-Depth. Train-only, not part of the checkpoint."""

    def __init__(self, *, moge_path: str, morgbd_path: str) -> None:
        moge_cls, morgbd_cls = _import_depth_teachers()
        self.moge = moge_cls.from_pretrained(str(moge_path)).eval()
        self.morgbd = morgbd_cls.from_pretrained(str(morgbd_path)).eval()
        for model in (self.moge, self.morgbd):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def to(self, device: torch.device) -> "FrozenDepthTeacher":
        self.moge.to(device)
        self.morgbd.to(device)
        return self

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or int(images.shape[2]) != 3:
            raise ValueError(f"Depth teacher images must be [B, V, 3, H, W], got {tuple(images.shape)}.")
        flat = images.reshape(-1, *images.shape[-3:]).float()
        if flat.dtype != torch.float32:
            flat = flat.float()
        if flat.max().detach() > 1.5:
            flat = flat / 255.0
        flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        autocast_enabled = flat.is_cuda
        with torch.autocast(device_type=flat.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            moge_out = self.moge.infer(
                flat, resolution_level=3, num_tokens=256, apply_mask=False
            )
            depth = moge_out["depth"].squeeze().detach().clone()
            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            features, _cls = self.morgbd.infer_feat(
                flat,
                depth,
                depth_down_scale=1,
                resolution_level=3,
                num_tokens=256,
                enable_depth_mask=False,
            )
        features = features.permute(0, 2, 3, 1).contiguous()
        return features.view(features.shape[0], -1, features.shape[-1])


def _import_utils3d() -> None:
    try:
        import utils3d  # noqa: F401
        return
    except ImportError:
        pass
    candidates = []
    env_root = os.environ.get("LAWAM_UTILS3D_ROOT", "")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path("/data/home/zyh/miniconda3/envs/lingbotvla/lib/python3.12/site-packages"))
    for parent in candidates:
        if not (parent / "utils3d").is_dir():
            continue
        sys.path.insert(0, str(parent))
        try:
            import utils3d  # noqa: F401
            return
        except ImportError:
            continue
        finally:
            if sys.path and sys.path[0] == str(parent):
                sys.path.pop(0)
    raise ImportError(
        "MoGe needs utils3d. It is not installed in this environment. "
        "Point LAWAM_UTILS3D_ROOT at a directory whose child is the utils3d package, "
        "or keep the existing lingbotvla checkout that already contains it."
    )


class DepthTargetStore:
    """Episode files written by scripts/precompute_depth_targets.py."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Depth teacher features are missing at "
                f"{manifest_path}. Generate them first with the lawam environment:\n"
                "  /data/home/zyh/miniconda3/envs/lawam/bin/python "
                "scripts/precompute_depth_targets.py"
            )
        self.manifest = json.loads(manifest_path.read_text())
        if not bool(self.manifest.get("complete", False)):
            raise RuntimeError(
                f"Depth targets at {self.root} are incomplete. Finish "
                "scripts/precompute_depth_targets.py before training."
            )
        views = tuple(self.manifest.get("views", ()))
        if views != DEPTH_VIEW_NAMES:
            raise ValueError(f"Depth manifest views must be {DEPTH_VIEW_NAMES}, got {views}.")
        if int(self.manifest.get("tokens", -1)) != DEPTH_TARGET_TOKENS:
            raise ValueError("Depth manifest token count is not 256.")
        if int(self.manifest.get("channels", -1)) != DEPTH_TARGET_DIM:
            raise ValueError("Depth manifest channel count is not 1024.")
        self._episode_id: int | None = None
        self._features: torch.Tensor | None = None

    def frame(self, episode_index: int, frame_index: int) -> torch.Tensor:
        episode_index = int(episode_index)
        frame_index = int(frame_index)
        if self._episode_id != episode_index or self._features is None:
            path = self.root / f"episode_{episode_index:06d}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing depth target file {path}.")
            features = torch.load(path, map_location="cpu", weights_only=True)
            if features.ndim != 4 or tuple(features.shape[1:]) != (len(DEPTH_VIEW_NAMES), DEPTH_TARGET_TOKENS, DEPTH_TARGET_DIM):
                raise ValueError(f"Unexpected depth target shape in {path}: {tuple(features.shape)}.")
            self._episode_id = episode_index
            self._features = features
        if frame_index < 0 or frame_index >= int(self._features.shape[0]):
            raise IndexError(
                f"Depth target frame {frame_index} is outside episode {episode_index} "
                f"length {int(self._features.shape[0])}."
            )
        return self._features[frame_index]


def _import_depth_teachers():
    try:
        from mdm.model.v2 import MDMModel
        from moge.model.v2 import MoGeModel
        return MoGeModel, MDMModel
    except ImportError:
        pass
    root = Path(os.environ.get("LAWAM_DEPTH_TEACHER_ROOT", DEFAULT_DEPTH_TEACHER_ROOT))
    moge_root = root / "MoGe"
    mdm_root = root / "lingbot-depth"
    if not moge_root.is_dir() or not mdm_root.is_dir():
        raise ImportError(
            "Depth teacher source trees were not found at "
            f"{moge_root} and {mdm_root}. Set LAWAM_DEPTH_TEACHER_ROOT to the "
            "lingbot-vla vision_models directory. They are imported from that checkout "
            "and are not installed into the lawam environment."
        )
    _import_utils3d()
    for path in (str(moge_root), str(mdm_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from mdm.model.v2 import MDMModel
        from moge.model.v2 import MoGeModel
    except Exception as exc:
        raise ImportError(
            "Could not import MoGe-2 and LingBot-Depth from the lingbot-vla checkout. "
            "Leave those packages out of the lawam environment install, and make sure this "
            f"interpreter can import their Python dependencies. Original error: {exc}"
        ) from exc
    return MoGeModel, MDMModel
