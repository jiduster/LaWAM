"""Task-conditioned compact geometry tokens; no future frames or labels in visual inference."""
import math
import torch
from torch import nn
from torch.nn import functional as F


MODES = ('none', 'oracle', 'visual', 'visual_aux')


class QueryBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.qnorm = nn.LayerNorm(dim)
        self.knorm = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.snorm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, query, context, valid=None):
        kv = self.knorm(context)
        query = query + self.cross(self.qnorm(query), kv, kv,
                                   key_padding_mask=None if valid is None else ~valid, need_weights=False)[0]
        q = self.snorm(query)
        query = query + self.self_attention(q, q, q, need_weights=False)[0]
        return query + self.ff(query)


class InteractionTokens(nn.Module):
    def __init__(self, *, mode, vision_dim, vlm_dim, state_dim, entities, dim=128,
                 queries=8, heads=8, layers=2, feature_dim=17, mean=None, std=None):
        super().__init__()
        if mode not in MODES[1:] or dim <= 0 or heads <= 0 or dim % heads or min(queries,layers,entities) < 1:
            raise ValueError('Invalid interaction token architecture')
        self.mode, self.entities, self.feature_dim = mode, entities, feature_dim
        self.query = nn.Parameter(torch.randn(1, queries, dim)*.02)
        self.entity_embedding = nn.Embedding(entities, dim)
        self.state = nn.Linear(state_dim*2, dim)
        self.language = nn.Linear(vlm_dim, dim)
        self.blocks = nn.ModuleList([QueryBlock(dim,heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        if mode == 'oracle':
            self.input_projection = nn.Linear(feature_dim*2,dim)
        else:
            self.input_projection = nn.Linear(vision_dim,dim)
            self.spatial_projection = nn.Sequential(nn.Linear(2,dim),nn.GELU(),nn.Linear(dim,dim))
        # Both visual controls have the SAME readout and parameter initialization.
        self.readout_attention = nn.MultiheadAttention(dim,heads,batch_first=True)
        self.readout = nn.Linear(dim,feature_dim)
        mean = torch.zeros(feature_dim) if mean is None else torch.as_tensor(mean,dtype=torch.float32)
        std = torch.ones(feature_dim) if std is None else torch.as_tensor(std,dtype=torch.float32)
        if mean.shape != (feature_dim,) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Interaction normalization must be finite train-only vectors with positive std')
        self.register_buffer('feature_mean',mean)
        self.register_buffer('feature_std',std)

    def normalize(self, value):
        return (value.float()-self.feature_mean.float())/self.feature_std.float()

    def forward(self, *, current_vision, vlm, language_mask, state, state_mask,
                geometry=None, geometry_mask=None):
        dtype = self.query.dtype
        if vlm.ndim != 3 or language_mask.shape != vlm.shape[:2]:
            raise ValueError('Interaction language mask must match the current VLM sequence')
        if state.ndim != 2 or state_mask.shape != state.shape or not language_mask.bool().any(-1).all():
            raise ValueError('Interaction requires masked current state and nonempty VLM context')
        lm = language_mask.to(vlm.dtype).unsqueeze(-1)
        language = (vlm*lm).sum(1)/lm.sum(1).clamp_min(1)
        state_valid = state_mask.to(state.dtype)
        condition = self.language(language.to(dtype)) + self.state(
            torch.cat((state*state_valid,state_valid),-1).to(dtype))
        query = self.query.expand(state.shape[0],-1,-1) + condition.unsqueeze(1)
        if self.mode == 'oracle':
            if geometry is None or geometry_mask is None:
                raise KeyError('Oracle interaction requires current interaction_geometry and interaction_valid')
            if geometry.shape != (state.shape[0],self.entities,self.feature_dim) or geometry_mask.shape != geometry.shape:
                raise ValueError('Oracle interaction geometry shape mismatch')
            valid = geometry_mask.bool().any(-1)
            if not valid.any(-1).all():
                raise ValueError('Oracle interaction requires at least one valid entity per sample')
            if not torch.isfinite(geometry[geometry_mask.bool()]).all():
                raise ValueError('Nonfinite valid interaction geometry')
            safe = torch.where(geometry_mask.bool(),geometry,torch.zeros_like(geometry))
            x = torch.cat((self.normalize(safe)*geometry_mask,geometry_mask.float()),-1)
            context = self.input_projection(x.to(dtype)) + self.entity_embedding.weight.unsqueeze(0)
        else:
            # Labels cannot enter this branch, even when supplied for auxiliary loss.
            count=current_vision.shape[1]
            side=math.isqrt(count)
            if side*side != count:
                raise ValueError('Interaction visual tokens require a square current DINO patch grid')
            axis=torch.linspace(-1,1,side,device=current_vision.device,dtype=dtype)
            yy,xx=torch.meshgrid(axis,axis,indexing='ij')
            xy=torch.stack((xx,yy),-1).reshape(1,count,2)
            context=self.input_projection(current_vision.to(dtype))+self.spatial_projection(xy)
            valid=None
        for block in self.blocks:
            query=block(query,context,valid)
        tokens=self.norm(query)
        entity_query=self.entity_embedding.weight.unsqueeze(0).expand(tokens.shape[0],-1,-1)
        prediction=self.readout(self.readout_attention(entity_query,tokens,tokens,need_weights=False)[0])
        return tokens,prediction

    def auxiliary_loss(self,prediction,target,valid):
        if target.shape != prediction.shape or valid.shape != prediction.shape:
            raise ValueError('Interaction auxiliary target/mask shape mismatch')
        valid=valid.bool()
        if not valid.any() or not torch.isfinite(target[valid]).all():
            raise ValueError('Interaction supervision requires finite valid targets')
        safe=torch.where(valid,target,torch.zeros_like(target))
        loss=F.smooth_l1_loss(prediction.float(),self.normalize(safe),reduction='none')
        return loss[valid].mean()
