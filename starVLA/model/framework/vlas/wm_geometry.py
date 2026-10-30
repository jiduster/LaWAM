"""Training-only object-state readout from predicted future DINO patches."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from starVLA.interaction_geometry import ENTITY_NAMES, FEATURE_DIM
from .interaction_tokens import PRED_VISUAL_MODES, QueryBlock

WM_MODES = ('visual_wm_aux', 'visual_wm_aux_detach')
FUTURE_GEOMETRY_MODES = (*WM_MODES, *PRED_VISUAL_MODES)


def dynamic_geometry_mask(task):
    """Compact task-relevant states; exclude static entities and symmetric rotations."""
    mask = torch.zeros(len(ENTITY_NAMES[task]), FEATURE_DIM, dtype=torch.bool)
    if task == 'bimanual_hanoi':
        mask[1:4, :3] = True  # large / medium / small disk world position
    elif task == 'bimanual_microwave_cook':
        mask[0, :3] = True  # food world position
        mask[2, 15] = True  # door hinge angle (not duplicated at the handle)
    else:
        raise ValueError(f'Unsupported future geometry task: {task}')
    return mask


class FutureGeometryHead(nn.Module):
    def __init__(self, *, vision_dim, task, dim=128, heads=8, layers=2,
                 mean, std, detach=False):
        super().__init__()
        self.detach = bool(detach)
        self.query = nn.Parameter(torch.randn(1, len(ENTITY_NAMES[task]), dim) * .02)
        self.input_projection = nn.Linear(vision_dim, dim)
        self.spatial_projection = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([QueryBlock(dim, heads) for _ in range(layers)])
        self.readout = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, FEATURE_DIM))
        mean, std = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)
        if (mean.shape != (FEATURE_DIM,) or std.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any()):
            raise ValueError('Future geometry requires finite train-only mean/std with positive std')
        self.register_buffer('feature_mean', mean)
        self.register_buffer('feature_std', std)
        self.register_buffer('supervision_mask', dynamic_geometry_mask(task))

    def forward(self, future):
        if future.ndim != 3:
            raise ValueError('Future geometry requires predicted future patches [B,K,D]')
        future = future.detach() if self.detach else future
        count = future.shape[1]
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError('Future geometry expects a square DINO patch grid')
        axis = torch.linspace(-1, 1, side, device=future.device, dtype=self.query.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        xy = torch.stack((xx, yy), -1).reshape(1, count, 2)
        context = self.input_projection(future.to(self.query.dtype)) + self.spatial_projection(xy)
        query = self.query.expand(future.shape[0], -1, -1)
        for block in self.blocks:
            query = block(query, context)
        return self.readout(query)

    def loss(self, prediction, target, valid):
        if target.shape != prediction.shape or valid.shape != target.shape:
            raise ValueError('Future geometry target/mask shape mismatch')
        valid = valid.bool() & self.supervision_mask
        if not torch.isfinite(target[valid]).all():
            raise ValueError('Nonfinite valid future geometry targets')
        # A batch can contain only padded endpoints. Keep the graph for DDP.
        safe = torch.where(valid, target.float(), self.feature_mean)
        normalized = (safe - self.feature_mean) / self.feature_std
        errors = F.smooth_l1_loss(prediction.float(), normalized, reduction='none')
        loss = torch.where(valid, errors, 0.).sum() / valid.sum().clamp_min(1)
        with torch.no_grad():
            physical = prediction.float() * self.feature_std + self.feature_mean
            error = physical - safe
            position_valid = valid[..., :3]
            position_rmse = (torch.where(position_valid, error[..., :3].square(), 0.).sum()
                             / position_valid.sum().clamp_min(1)).sqrt()
            angle_mae = (torch.where(valid[..., 15], error[..., 15].abs(), 0.).sum()
                         / valid[..., 15].sum().clamp_min(1))
        return loss, position_rmse, angle_mae
