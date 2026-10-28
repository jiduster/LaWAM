from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_action_model.core.lam_model import load_latent_action_model
from starVLA.model.framework.latent_world.processor_utils import (
    build_latent_world_processor_spec,
    configure_latent_world_processor,
)
from starVLA.model.framework.latent_world.types import LatentWorldPolicyInferBatch, LatentWorldPolicyTrainBatch
from .vlm_auto import (
    load_vlm_auto,
    remove_lm_head,
)
from starVLA.model.tools import sync_managed_modules_training_mode

from .depth_queries import (
    TaskTokenDepthHead,
    build_depth_query_attention_mask,
    depth_placeholder_spans,
    gather_depth_contexts,
    image_token_spans,
)
from .flowmatching_expert import ConditionalFlowMatchingConfig, ConditionalFlowMatchingHead
from .latent_action_head import LatentActionHead as VLMToLAMQFormer
from .pose_conditioning import PoseEncoder
from .interaction_tokens import InteractionTokens, MODES
from .wm_geometry import FutureGeometryHead, WM_MODES


# ============================================================================
# Config & Constants
# ============================================================================

DEFAULT_LAM_ROOT = Path("latent_action_model")


@dataclass
class LatentWorldPolicyConfig:
    """Independent LatentWorldPolicyBackend config."""

    # Flow head config
    flow_cfg: ConditionalFlowMatchingConfig = field(default_factory=ConditionalFlowMatchingConfig)

    # Action chunk (Qwen-compatible schema)
    future_action_window_size: int = 7
    past_action_window_size: int = 0
    action_horizon: int = 8

    # Base checkpoints
    hf_cache_dir: Optional[Union[str, Path]] = None
    lam_ckpt_path: str = str(DEFAULT_LAM_ROOT / "logs/dino_base_ae_bridge/version_0/checkpoints/epoch=39.ckpt")
    lam_yaml_path: str = str(DEFAULT_LAM_ROOT / "logs/dino_base_ae_bridge/version_0/dino_base_ae.yaml")

    # VLM dtype
    vlm_dtype: torch.dtype = torch.bfloat16
    remove_lm_head: bool = True

    # Placeholder token
    latent_action_placeholder_token: str = "<ACT_PH>"

    # World / distill losses
    perceptual_weight: float = 0.1
    enable_loss_distill: bool = True
    latent_loss_type: str = "mse"  # "mse" | "cosine"
    lam_encoder_distill_weight: float = 1.0

    # Flow conditioning variants
    future_prediction: bool = False
    repeated_diffusion_steps: int = 4
    flow_only_mode: bool = False
    enable_flow_h_t1_scheduled_sampling: bool = False
    flow_h_t1_pred_prob_start: float = 0.0
    flow_h_t1_pred_prob_end: float = 1.0
    flow_h_t1_pred_ramp_steps: int = 20000
    detach_future_feature: bool = False
    # "predicted": flow reads the world model's future DINO tokens.
    # "current": that slot is a copy of the current-frame DINO tokens. Perceptual
    # and distill losses still train the world model; only the action condition changes.
    flow_future_feature: str = "predicted"

    # Independent additions
    num_action_queries: int = 8
    flow_action_num_queries: int = 8
    latent_action_head_type: str = "vlm_queries"  # "vlm_queries" | "external_qformer"

    # Current action conditioning or training-only visual_wm_aux[_detach].
    interaction_mode: str = "none"
    interaction_task: str = ""
    interaction_entities: int = 0
    interaction_feature_dim: int = 17
    interaction_queries: int = 8
    interaction_layers: int = 2
    interaction_loss_weight: float = 0.0
    interaction_sidecar_root: str = ""
    interaction_feature_mean: Optional[Tuple[float, ...]] = None
    interaction_feature_std: Optional[Tuple[float, ...]] = None

    # Green branch: current pose adjusts private planning queries before the LAM latent.
    plan_pose_conditioning: bool = False
    plan_pose_token_dim: int = 128
    plan_pose_hidden_dim: int = 128
    plan_pose_attention_heads: int = 8
    plan_pose_position_mean: Optional[Tuple[float, ...]] = None
    plan_pose_position_std: Optional[Tuple[float, ...]] = None

    # DexJoCo fingertip auxiliary prediction. The head is only constructed when
    # the loss weight is positive, so existing checkpoints remain compatible.
    # "delta": full-horizon displacement from the state-conditioned future-DINO head.
    # "endpoint": 8 absolute positions at the last action-chunk frame, from h_t1_pred.
    # "latent_delta": Stage 1 loss moved onto the Stage 2 latent. The head reads the
    # predicted latent and regresses the 8-tip displacement across that same chunk.
    fingertip_loss_weight: float = 0.0
    contact_sidecar_root: str = ""
    object_pose_sidecar_root: str = ""
    action_expert_only: bool = False
    fingertip_num_points: int = 8
    fingertip_target_mode: str = "delta"
    fingertip_head_hidden_dim: int = 512
    fingertip_target_scale: float = 0.1
    fingertip_target_center: tuple[float, float, float] = (0.0, 0.0, 1.3)

    # LingBot query-mode depth alignment. The 24 VLM hidden states stay in h_vlm
    # for the flow. The 256 x teacher_dim head is a smooth-L1 target only.
    depth_query_conditioning: bool = False
    depth_placeholder_token: str = "<DEPTH_PH>"
    depth_num_views: int = 3
    depth_num_task_tokens: int = 8
    depth_num_backbone_tokens: int = 256
    depth_teacher_dim: int = 1024
    depth_loss_weight: float = 0.004
    depth_feature_root: str = ""
    depth_head_layers: int = 1
    depth_head_heads: int = 4
    depth_head_dim_head: int = 32
    depth_head_ff_mult: int = 1
    moge_path: str = ""
    morgbd_path: str = ""

    @property
    def num_vlm_action_queries(self) -> int:
        head_type = str(self.latent_action_head_type).lower()
        if head_type not in {"vlm_queries", "external_qformer"}:
            raise ValueError(f"Unknown latent_action_head_type: {self.latent_action_head_type!r}.")
        if head_type == "external_qformer":
            if self.flow_only_mode:
                raise ValueError("external_qformer requires flow_only_mode=false; no VLM act-query boundary exists.")
            return 0
        count = int(self.num_action_queries)
        if count <= 0:
            raise ValueError("vlm_queries requires num_action_queries > 0.")
        return count


# ============================================================================
# VLM / LAM Load Helpers
# ============================================================================


def _load_vlm_and_processor(
    cfg: LatentWorldPolicyConfig,
    vlm_model_id: str,
) -> Tuple[nn.Module, Any, Any, int]:
    processor_spec = build_latent_world_processor_spec(policy_cfg=cfg, vlm_model_id=vlm_model_id)
    vlm, processor = load_vlm_auto(processor_spec.model_id, processor_spec.cache_dir, dtype=cfg.vlm_dtype)
    processor, tokenizer, placeholder_token_id, depth_token_id = configure_latent_world_processor(
        processor,
        placeholder_token=processor_spec.placeholder_token,
        depth_placeholder_token=processor_spec.depth_placeholder_token,
    )

    # Ensure tokenizer size and embedding rows stay aligned after adding placeholder tokens.
    target_vocab_size = max(int(len(tokenizer)), int(placeholder_token_id + 1))
    if depth_token_id is not None:
        target_vocab_size = max(target_vocab_size, int(depth_token_id) + 1)
    embed = vlm.get_input_embeddings()
    if embed is None:
        raise RuntimeError("[LatentWorldPolicyBackend] VLM does not expose input embeddings.")
    embed_rows = int(embed.weight.shape[0])
    if target_vocab_size > embed_rows:
        if not hasattr(vlm, "resize_token_embeddings"):
            raise RuntimeError(
                "[LatentWorldPolicyBackend] tokenizer vocab exceeds embedding size, "
                "but model does not support `resize_token_embeddings`."
            )
        vlm.resize_token_embeddings(target_vocab_size)

    if hasattr(vlm, "generation_config") and vlm.generation_config is not None:
        vlm.generation_config.max_new_tokens = 4
    if hasattr(vlm, "config") and vlm.config is not None:
        try:
            vlm.config.loss_type = "ForCausalLMLoss"
            vlm.config.use_cache = False
        except Exception:
            pass

    if bool(getattr(cfg, "remove_lm_head", True)):
        remove_lm_head(vlm)

    return vlm, processor, tokenizer, placeholder_token_id, depth_token_id


def _apply_flow_only_grad_to_h_vlm(
    *,
    h_vlm: torch.Tensor,
    act_placeholder_mask: torch.Tensor,
    enable_flow_only: bool,
) -> torch.Tensor:
    """
    Keep full VLM context values for flow, but split direct-flow gradients by sequence boundary:
    only tokens after the last act placeholder keep direct flow gradients.
    """
    if not bool(enable_flow_only):
        return h_vlm
    if h_vlm.dim() != 3:
        raise ValueError(f"[LatentWorldPolicyBackend] expected h_vlm [B, L, D], got {tuple(h_vlm.shape)}")
    if act_placeholder_mask is None:
        raise ValueError("[LatentWorldPolicyBackend] flow_only_mode=True requires act_placeholder_mask.")
    if act_placeholder_mask.dim() != 2:
        raise ValueError(
            f"[LatentWorldPolicyBackend] expected act_placeholder_mask [B, L], got {tuple(act_placeholder_mask.shape)}"
        )
    if act_placeholder_mask.shape[0] != h_vlm.shape[0] or act_placeholder_mask.shape[1] != h_vlm.shape[1]:
        raise ValueError(
            "[LatentWorldPolicyBackend] act_placeholder_mask shape mismatch with h_vlm: "
            f"mask={tuple(act_placeholder_mask.shape)} vs h_vlm={tuple(h_vlm.shape)}"
        )
    act_mask = act_placeholder_mask.to(device=h_vlm.device, dtype=torch.bool)
    if not torch.all(act_mask.any(dim=1)):
        raise ValueError("[LatentWorldPolicyBackend] flow_only_mode=True requires at least one act placeholder per sample.")

    seq_positions = torch.arange(h_vlm.shape[1], device=h_vlm.device).unsqueeze(0)
    last_act_idx = torch.where(act_mask, seq_positions, -1).max(dim=1).values
    post_act_mask = seq_positions > last_act_idx.unsqueeze(1)
    post_act_mask_f = post_act_mask.unsqueeze(-1).to(dtype=h_vlm.dtype)
    return h_vlm * post_act_mask_f + h_vlm.detach() * (1.0 - post_act_mask_f)


def _module_param_dtype(module: Optional[nn.Module], default: torch.dtype) -> torch.dtype:
    if module is None:
        return default
    try:
        p = next(module.parameters())
        return p.dtype
    except Exception:
        return default


def _cuda_autocast(dtype: torch.dtype):
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


@dataclass
class PolicyEncodingState:
    h_vlm: torch.Tensor
    pred_action_emb: torch.Tensor
    h_t: torch.Tensor
    h_t1_pred: torch.Tensor
    h_t1_gt: torch.Tensor
    h_t_original: torch.Tensor
    interaction_tokens: Optional[torch.Tensor] = None
    interaction_prediction: Optional[torch.Tensor] = None


# ============================================================================
# LatentWorldPolicyBackend
# ============================================================================


class LatentWorldPolicyBackend(nn.Module):
    """Independent world-model VLA without LatentVLAModel dependency."""

    def __init__(self, model_cfg: LatentWorldPolicyConfig, *, vlm_model_id: str) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.vlm_model_id = str(vlm_model_id)
        self.num_action_queries = int(self.model_cfg.num_vlm_action_queries)

        # 1) Load VLM + processor/tokenizer and register placeholder token.
        (
            self.vlm,
            self.processor,
            self.tokenizer,
            self.placeholder_token_id,
            self.depth_placeholder_token_id,
        ) = _load_vlm_and_processor(
            self.model_cfg,
            self.vlm_model_id,
        )

        # 2) Load LAM.
        self.lam = load_latent_action_model(self.model_cfg.lam_ckpt_path, self.model_cfg.lam_yaml_path)

        # 3) Create trainable query and mapping head.
        vlm_cfg = getattr(self.vlm, "config", None)
        text_cfg = getattr(vlm_cfg, "text_config", None)
        # NOTE: don't nest `getattr` in default value; default expression is evaluated eagerly.
        vlm_hidden_size = getattr(text_cfg, "hidden_size", None)
        if vlm_hidden_size is None:
            vlm_hidden_size = getattr(vlm_cfg, "hidden_size", None)
        if vlm_hidden_size is None:
            raise AttributeError("[LatentWorldPolicyBackend] cannot resolve VLM hidden_size from config.")
        vlm_hidden_dim = int(vlm_hidden_size)
        lam_code_dim = int(self.lam.code_dim)
        self.act_query = (
            nn.Parameter(torch.randn(self.num_action_queries, vlm_hidden_dim) * 0.02)
            if self.num_action_queries > 0 else None
        )
        self.flow_action_num_queries = int(self.model_cfg.flow_action_num_queries)
        self.flow_action_query = nn.Parameter(torch.randn(self.flow_action_num_queries, vlm_hidden_dim) * 0.02)
        self.depth_align_head: Optional[TaskTokenDepthHead] = None
        if bool(self.model_cfg.depth_query_conditioning):
            if self.depth_placeholder_token_id is None:
                raise RuntimeError("Depth query conditioning requires the <DEPTH_PH> token.")
            num_views = int(self.model_cfg.depth_num_views)
            num_task = int(self.model_cfg.depth_num_task_tokens)
            if num_views <= 0 or num_task <= 0:
                raise ValueError("depth_num_views and depth_num_task_tokens must be positive.")
            feature_root = str(self.model_cfg.depth_feature_root).strip()
            if float(self.model_cfg.depth_loss_weight) > 0.0 and not feature_root:
                raise ValueError(
                    "depth_loss_weight > 0 requires framework.action_model.depth_feature_root. "
                    "Precompute it with scripts/precompute_depth_targets.py in the lawam environment."
                )
            self.depth_align_head = TaskTokenDepthHead(
                vlm_dim=vlm_hidden_dim,
                teacher_dim=int(self.model_cfg.depth_teacher_dim),
                num_backbone_tokens=int(self.model_cfg.depth_num_backbone_tokens),
                num_task_tokens=num_task,
                num_layers=int(self.model_cfg.depth_head_layers),
                num_heads=int(self.model_cfg.depth_head_heads),
                dim_head=int(self.model_cfg.depth_head_dim_head),
                ff_mult=int(self.model_cfg.depth_head_ff_mult),
            )
            language_model = getattr(getattr(self.vlm, "model", None), "language_model", None)
            if language_model is not None and getattr(language_model.config, "_attn_implementation", "sdpa") not in {
                "sdpa",
                "eager",
            }:
                language_model.config._attn_implementation = "sdpa"
            print(
                "[LatentWorldPolicyBackend] depth queries enabled: "
                f"{num_views} views x {num_task} tokens, "
                f"backbone={int(self.model_cfg.depth_num_backbone_tokens)}, "
                f"depth_loss_weight={float(self.model_cfg.depth_loss_weight)}, "
                f"action_horizon={int(self.model_cfg.action_horizon)}, "
                f"depth_feature_root={feature_root}. "
                "Depth-query hidden states stay in h_vlm for the flow. "
                "The depth head output is loss-only and is matched to precomputed teacher features."
            )
        # Keep the pretrained parameter prefix in both modes for exact head initialization.
        self.vlm_to_lam = VLMToLAMQFormer(
            vlm_hidden_dim=vlm_hidden_dim,
            lam_code_dim=lam_code_dim,
            num_layers=1,
            num_heads=8,
            ffn_expansion_factor=4.0,
            dropout=0.0,
            pose_conditioning=bool(self.model_cfg.plan_pose_conditioning),
            pose_fusion="context" if self.num_action_queries > 0 else "query",
            pose_token_dim=int(self.model_cfg.plan_pose_token_dim),
            pose_attention_heads=int(self.model_cfg.plan_pose_attention_heads),
        )
        self.plan_pose_encoder = None
        if self.model_cfg.plan_pose_conditioning:
            if self.model_cfg.action_expert_only:
                raise ValueError("plan_pose_conditioning requires action_expert_only=false and planning losses")
            if not self.model_cfg.future_prediction or self.model_cfg.flow_future_feature != "predicted":
                raise ValueError("Planning pose requires future_prediction=true and flow_future_feature=predicted")
            if self.model_cfg.plan_pose_position_mean is None or self.model_cfg.plan_pose_position_std is None:
                raise ValueError("Planning pose requires training-only plan_pose_position_mean/std")
            with torch.random.fork_rng(devices=[]):
                self.plan_pose_encoder = PoseEncoder(
                    int(self.model_cfg.plan_pose_token_dim), int(self.model_cfg.plan_pose_hidden_dim),
                    self.model_cfg.plan_pose_position_mean, self.model_cfg.plan_pose_position_std,
                )

        self.fingertip_head: Optional[nn.Module] = None
        self._fingertip_target_mode = str(self.model_cfg.fingertip_target_mode).lower()
        if float(self.model_cfg.fingertip_loss_weight) > 0.0:
            if int(self.model_cfg.fingertip_num_points) <= 0:
                raise ValueError("fingertip_num_points must be positive when fingertip loss is enabled.")
            if self._fingertip_target_mode not in {"delta", "endpoint", "latent_delta"}:
                raise ValueError(
                    "fingertip_target_mode must be 'delta', 'endpoint', or 'latent_delta', "
                    f"got {self.model_cfg.fingertip_target_mode!r}."
                )
            if self._fingertip_target_mode == "endpoint" and not bool(self.model_cfg.future_prediction):
                raise ValueError("fingertip_target_mode='endpoint' requires future_prediction=true.")
            if self._fingertip_target_mode in {"endpoint", "latent_delta"} and float(self.model_cfg.fingertip_target_scale) <= 0.0:
                raise ValueError("fingertip_target_scale must be positive for this fingertip loss.")
            hidden_dim = int(self.model_cfg.fingertip_head_hidden_dim)
            if hidden_dim <= 0:
                raise ValueError("fingertip_head_hidden_dim must be positive.")
            num_points = int(self.model_cfg.fingertip_num_points)
            if self._fingertip_target_mode == "latent_delta":
                # Same readout as Stage 1: one displacement vector from the latent code.
                in_dim = int(self.lam.code_dim)
                out_dim = num_points * 3
            elif self._fingertip_target_mode == "endpoint":
                in_dim = int(self.lam.input_dim)
                out_dim = num_points * 3
            else:
                in_dim = int(self.lam.input_dim) + int(self.model_cfg.flow_cfg.state_dim)
                out_dim = int(self.model_cfg.action_horizon) * num_points * 3
            self.fingertip_head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

        # 4) Align flow config to LAM output dimensions.
        lam_vision_dim = int(self.lam.input_dim)
        lam_grid_h = int(getattr(self.lam.encoder, "grid_height", 0) or 0)
        lam_grid_w = int(getattr(self.lam.encoder, "grid_width", 0) or 0)
        lam_num_tokens = int(lam_grid_h * lam_grid_w) if lam_grid_h > 0 and lam_grid_w > 0 else int(
            self.model_cfg.flow_cfg.num_vision_tokens
        )
        if int(self.model_cfg.flow_cfg.vision_dim) != lam_vision_dim:
            print(
                f"[LatentWorldPolicyBackend] flow_cfg.vision_dim={self.model_cfg.flow_cfg.vision_dim} "
                f"!= LAM vision dim={lam_vision_dim}; auto-aligned."
            )
        self.model_cfg.flow_cfg.vision_dim = lam_vision_dim

        if int(self.model_cfg.flow_cfg.num_vision_tokens) != lam_num_tokens:
            print(
                f"[LatentWorldPolicyBackend] flow_cfg.num_vision_tokens={self.model_cfg.flow_cfg.num_vision_tokens} "
                f"!= LAM token count={lam_num_tokens}; auto-aligned."
            )
            self.model_cfg.flow_cfg.num_vision_tokens = lam_num_tokens

        # 5) Flow head.
        self.flow = ConditionalFlowMatchingHead(config=self.model_cfg.flow_cfg)
        self.flow.action_horizon = int(self.model_cfg.action_horizon)
        self._flow_train_step: int = 0
        flow_future_feature = str(self.model_cfg.flow_future_feature).lower()
        if flow_future_feature not in {"predicted", "current"}:
            raise ValueError(
                "flow_future_feature must be 'predicted' or 'current', "
                f"got {self.model_cfg.flow_future_feature!r}."
            )
        self.model_cfg.flow_future_feature = flow_future_feature
        if flow_future_feature == "current":
            print(
                "[LatentWorldPolicyBackend] flow future feature is the current-frame DINO tokens. "
                "Perceptual and distill losses still use the predicted future."
            )
        if bool(self.model_cfg.enable_flow_h_t1_scheduled_sampling) and not bool(self.model_cfg.detach_future_feature):
            print(
                "[LatentWorldPolicyBackend][warn] enable_flow_h_t1_scheduled_sampling=true and "
                "detach_future_feature=false: flow gradients through GT-conditioned branch can increase loss_perceptual."
            )
        if self.model_cfg.action_expert_only:
            self.configure_action_expert_only()
        self.sync_training_modes(mode=self.training)

        self.interaction_encoder = None
        if self.model_cfg.interaction_mode in MODES[1:]:
            with torch.random.fork_rng(devices=[]):
                self.interaction_encoder = InteractionTokens(
                    mode=self.model_cfg.interaction_mode,vision_dim=lam_vision_dim,
                    vlm_dim=vlm_hidden_dim,state_dim=int(self.model_cfg.flow_cfg.state_dim),
                    entities=int(self.model_cfg.interaction_entities),
                    dim=int(self.model_cfg.flow_cfg.interaction_token_dim),
                    heads=int(self.model_cfg.flow_cfg.interaction_heads),
                    queries=int(self.model_cfg.interaction_queries),layers=int(self.model_cfg.interaction_layers),
                    feature_dim=int(self.model_cfg.interaction_feature_dim),
                    mean=self.model_cfg.interaction_feature_mean,std=self.model_cfg.interaction_feature_std)
        self.wm_geometry_head = None
        if self.model_cfg.interaction_mode in WM_MODES:
            if self.model_cfg.action_expert_only or not self.model_cfg.future_prediction:
                raise ValueError('Future geometry supervision requires full Stage 2 with future_prediction=true')
            # Preserve policy initialization and diffusion RNG across the two controls.
            with torch.random.fork_rng(devices=[]):
                self.wm_geometry_head = FutureGeometryHead(
                    vision_dim=lam_vision_dim, task=self.model_cfg.interaction_task,
                    dim=int(self.model_cfg.flow_cfg.interaction_token_dim),
                    heads=int(self.model_cfg.flow_cfg.interaction_heads),
                    layers=int(self.model_cfg.interaction_layers),
                    mean=self.model_cfg.interaction_feature_mean, std=self.model_cfg.interaction_feature_std,
                    detach=self.model_cfg.interaction_mode == 'visual_wm_aux_detach')

    def _inject_queries(
        self,
        *,
        inputs_embeds: torch.Tensor,
        placeholder_mask: torch.BoolTensor,
        queries: torch.Tensor,
        num_queries: int,
        name: str,
    ) -> None:
        device = inputs_embeds.device
        placeholder_mask = placeholder_mask.to(device=device, dtype=torch.bool)

        bsz, seq_len = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
        expected = int(bsz * num_queries)
        got = int(placeholder_mask.sum().item())
        if got != expected:
            raise ValueError(
                f"[LatentWorldPolicyBackend] {name} placeholder count mismatch: got={got}, expected={expected}"
            )

        per_sample = placeholder_mask.sum(dim=1)
        if not torch.all(per_sample == int(num_queries)):
            bad = torch.nonzero(per_sample != int(num_queries), as_tuple=False).flatten()
            b = int(bad[0].item()) if bad.numel() > 0 else -1
            got_b = int(per_sample[b].item()) if b >= 0 else -1
            raise ValueError(
                f"[LatentWorldPolicyBackend] {name} placeholder count mismatch for sample {b}: got={got_b}, expected={num_queries}"
            )

        idx = placeholder_mask.nonzero(as_tuple=False)
        flat = idx[:, 0] * seq_len + idx[:, 1]
        idx = idx[flat.argsort()]
        b_idx, p_idx = idx[:, 0], idx[:, 1]
        q_idx = torch.arange(int(num_queries), device=device).repeat(bsz)
        inputs_embeds[b_idx, p_idx, :] = queries[q_idx]

    def _run_vlm_stage(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor],
        act_placeholder_mask: torch.BoolTensor,
        flow_placeholder_mask: torch.BoolTensor,
        act_query: Optional[torch.Tensor],
        flow_query: torch.Tensor,
        depth_placeholder_mask: Optional[torch.Tensor] = None,
        plan_pose_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        embed = self.vlm.get_input_embeddings()
        if embed is None:
            raise RuntimeError("[LatentWorldPolicyBackend] VLM does not expose input embeddings.")
        inputs_embeds = embed(input_ids)

        if self.num_action_queries > 0:
            if act_query is None:
                raise ValueError("VLM action queries are missing for latent_action_head_type='vlm_queries'.")
            self._inject_queries(
                inputs_embeds=inputs_embeds,
                placeholder_mask=act_placeholder_mask,
                queries=act_query,
                num_queries=int(self.num_action_queries),
                name="act_query",
            )
        elif act_placeholder_mask.any():
            raise ValueError("external_qformer batches must not contain VLM latent action placeholders.")
        self._inject_queries(
            inputs_embeds=inputs_embeds,
            placeholder_mask=flow_placeholder_mask,
            queries=flow_query,
            num_queries=int(flow_query.shape[0]),
            name="flow_query",
        )
        vlm_attention_mask: torch.Tensor = attention_mask
        position_ids = None
        if self.depth_align_head is not None:
            if depth_placeholder_mask is None or image_grid_thw is None:
                raise ValueError(
                    "Depth queries require depth_placeholder_mask and image_grid_thw."
                )
            task_tokens = self.depth_align_head.pooled_task_tokens().to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            view_count = int(self.model_cfg.depth_num_views)
            injected = task_tokens.repeat(view_count, 1)
            self._inject_queries(
                inputs_embeds=inputs_embeds,
                placeholder_mask=depth_placeholder_mask,
                queries=injected,
                num_queries=int(injected.shape[0]),
                name="depth_query",
            )
            depth_spans = depth_placeholder_spans(
                depth_placeholder_mask.to(device=inputs_embeds.device),
                tokens_per_view=int(self.model_cfg.depth_num_task_tokens),
            )
            views_per_sample = [len(spans) for spans in depth_spans]
            image_spans = image_token_spans(
                input_ids,
                image_token_id=int(self.vlm.config.image_token_id),
                image_grid_thw=image_grid_thw,
                merge_size=int(self.vlm.config.vision_config.spatial_merge_size),
                views_per_sample=views_per_sample,
            )
            vlm_attention_mask = build_depth_query_attention_mask(
                attention_mask, image_spans, depth_spans
            )
            position_ids = self.vlm.model.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
            if position_ids is None:
                raise RuntimeError("Qwen3-VL did not return 3D position ids for the depth-query mask.")

        # HF transformers 4.57 Qwen3VL causal wrapper drops `hidden_states`
        # from the returned output; use the underlying multimodal model output.
        vlm_out = self.vlm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=vlm_attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        if vlm_out is None:
            raise RuntimeError(
                "[LatentWorldPolicyBackend] Qwen3VL base model forward returned None; "
                "expected `last_hidden_state`."
            )
        hidden = getattr(vlm_out, "last_hidden_state", None)
        if hidden is None:
            output_keys = list(vlm_out.keys()) if hasattr(vlm_out, "keys") else None
            raise RuntimeError(
                "[LatentWorldPolicyBackend] Qwen3VL base model output does not expose "
                "`last_hidden_state`. "
                f"output_type={type(vlm_out).__name__}, output_keys={output_keys}."
            )
        if self.num_action_queries == 0:
            # Both heads read the same full VLM output, including wrist and text tokens.
            pred_latent = self.vlm_to_lam(hidden, attention_mask=attention_mask, pose_tokens=plan_pose_tokens)
        else:
            bsz = int(hidden.shape[0])
            q = int(self.num_action_queries)
            h_act = hidden[act_placeholder_mask]
            if h_act.numel() == 0:
                raise ValueError("[LatentWorldPolicyBackend] empty action hidden selection; check act_placeholder_mask.")
            if h_act.shape[0] != bsz * q:
                raise ValueError(
                    f"[LatentWorldPolicyBackend] action hidden count mismatch: got={h_act.shape[0]}, expected={bsz * q}"
                )
            pred_latent = self.vlm_to_lam(h_act.view(bsz, q, -1), pose_tokens=plan_pose_tokens)
        return {
            "h_vlm": hidden,
            "pred_latent": pred_latent,
            "vlm_out": vlm_out,
        }

    def _build_lam_teacher_inputs_for_distill(self, primary_video: torch.Tensor) -> torch.Tensor:
        expected_t = int(getattr(getattr(self.lam, "encoder", None), "num_frames", 0) or 0)
        if expected_t <= 0:
            return primary_video

        cur_t = int(primary_video.shape[1])
        if cur_t == expected_t:
            return primary_video
        if cur_t < expected_t:
            raise ValueError(
                f"[LatentWorldPolicyBackend] distill teacher temporal mismatch: got T={cur_t}, "
                f"but LAM encoder expects num_frames={expected_t}."
            )
        idx = torch.linspace(0, cur_t - 1, steps=expected_t, device=primary_video.device).round().long()
        return primary_video.index_select(1, idx)

    def _run_lam_teacher(self, *, primary_video: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        primary_video_t = self._build_lam_teacher_inputs_for_distill(primary_video)
        with torch.no_grad():
            lam_out = self.lam.get_latent_action(
                videos=primary_video_t,
                states=None,
                dec_videos=primary_video_t,
                predict_future_frame=False,
                embodiment_ids=embodiment_id,
            )
        # Some LAM implementations return inference tensors here. Materialize a normal
        # tensor before using it in training losses, otherwise autograd cannot save it.
        return lam_out["quantized"].detach().clone()

    def _compute_latent_loss(
        self,
        *,
        pred_latent: torch.Tensor,
        teacher_latent: torch.Tensor,
        latent_loss_type: str,
    ) -> torch.Tensor:
        if pred_latent.shape != teacher_latent.shape:
            raise ValueError(
                f"[LatentWorldPolicyBackend] latent shape mismatch: pred={tuple(pred_latent.shape)}, "
                f"teacher={tuple(teacher_latent.shape)}"
            )
        if str(latent_loss_type).lower() == "mse":
            return F.mse_loss(pred_latent, teacher_latent)
        return 1 - F.cosine_similarity(pred_latent, teacher_latent, dim=-1).mean()

    def _compute_distill_loss(
        self,
        *,
        pred_latent: torch.Tensor,
        primary_video: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        teacher_latent = self._run_lam_teacher(primary_video=primary_video, embodiment_id=embodiment_id)
        return self._compute_latent_loss(
            pred_latent=pred_latent,
            teacher_latent=teacher_latent,
            latent_loss_type=self.model_cfg.latent_loss_type,
        )

    def _compute_depth_loss(
        self,
        *,
        h_vlm: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
        depth_mask: torch.Tensor,
        depth_targets: torch.Tensor,
    ) -> torch.Tensor:
        if self.depth_align_head is None:
            raise RuntimeError("Depth loss requested without a depth head.")
        if image_grid_thw is None:
            raise ValueError("Depth loss requires image_grid_thw.")
        depth_spans = depth_placeholder_spans(
            depth_mask,
            tokens_per_view=int(self.model_cfg.depth_num_task_tokens),
        )
        image_spans = image_token_spans(
            input_ids,
            image_token_id=int(self.vlm.config.image_token_id),
            image_grid_thw=image_grid_thw,
            merge_size=int(self.vlm.config.vision_config.spatial_merge_size),
            views_per_sample=[len(spans) for spans in depth_spans],
        )
        context, context_mask = gather_depth_contexts(h_vlm, image_spans, depth_spans)
        prediction = self.depth_align_head(context, context_mask)
        target = depth_targets.reshape(-1, prediction.shape[1], prediction.shape[2])
        if tuple(prediction.shape) != tuple(target.shape):
            raise ValueError(
                "Depth prediction/target shape mismatch: "
                f"pred={tuple(prediction.shape)}, target={tuple(target.shape)}."
            )
        return F.smooth_l1_loss(prediction.float(), target.float().detach())

    def _decode_future_tokens_strict_single_query(
        self,
        *,
        h_t: torch.Tensor,
        pred_action_emb: torch.Tensor,
        source: str,
    ) -> torch.Tensor:
        if pred_action_emb.shape[1] != 1:
            raise ValueError(
                f"[{source}] future_prediction requires single-query latent action, "
                f"got query_dim={pred_action_emb.shape[1]}."
            )
        decoded = self.lam.decoder(h_t, pred_action_emb)
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        if decoded.dim() == 4:
            decoded = decoded[:, 0, :, :] if decoded.shape[1] == 1 else decoded[:, -1, :, :]
        return decoded

    @classmethod
    def build(
        cls,
        cfg: LatentWorldPolicyConfig,
        *,
        vlm_model_id: str,
    ) -> "LatentWorldPolicyBackend":
        model = cls(cfg, vlm_model_id=vlm_model_id)
        return model

    def configure_action_expert_only(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.flow.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.sync_training_modes(mode=mode)
        return self

    def sync_training_modes(self, mode: bool = True) -> None:
        sync_managed_modules_training_mode(
            self.vlm,
            self.lam,
            self.flow,
            self.vlm_to_lam,
            self.plan_pose_encoder,
            self.fingertip_head,
            self.depth_align_head,
            mode=mode,
        )

    def set_flow_train_step(self, step: int) -> None:
        self._flow_train_step = max(0, int(step))

    def _flow_h_t1_pred_prob(self) -> float:
        if not bool(self.model_cfg.enable_flow_h_t1_scheduled_sampling):
            return 1.0
        start = float(self.model_cfg.flow_h_t1_pred_prob_start)
        end = float(self.model_cfg.flow_h_t1_pred_prob_end)
        ramp_steps = max(1, int(self.model_cfg.flow_h_t1_pred_ramp_steps))
        ratio = min(1.0, float(self._flow_train_step) / float(ramp_steps))
        prob = start + (end - start) * ratio
        return float(max(0.0, min(1.0, prob)))

    def _build_flow_future_condition(
        self,
        *,
        h_t1_pred: torch.Tensor,
        h_t1_gt: torch.Tensor,
    ) -> torch.Tensor:
        if h_t1_pred.shape != h_t1_gt.shape:
            raise ValueError(
                f"[LatentWorldPolicyBackend] h_t1 shape mismatch: pred={tuple(h_t1_pred.shape)}, gt={tuple(h_t1_gt.shape)}"
            )

        if not self.training:
            return h_t1_pred

        prob_pred = self._flow_h_t1_pred_prob()
        bsz = int(h_t1_pred.shape[0])
        pred_mask = (torch.rand(bsz, 1, 1, device=h_t1_pred.device) < prob_pred)
        cond_future = torch.where(pred_mask, h_t1_pred, h_t1_gt)

        if not bool(self.model_cfg.detach_future_feature):
            gt_mask = (~pred_mask).to(dtype=h_t1_pred.dtype)
            # Straight-through bridge:
            cond_future = cond_future + gt_mask * (h_t1_pred - h_t1_pred.detach())
        else:
            cond_future = cond_future.detach()

        return cond_future

    def _select_flow_future_feature(
        self,
        *,
        h_t: torch.Tensor,
        h_t1_pred: torch.Tensor,
        h_t1_gt: torch.Tensor,
    ) -> torch.Tensor:
        """Choose the future-feature slot consumed by the action flow."""
        if str(self.model_cfg.flow_future_feature).lower() == "predicted":
            return self._build_flow_future_condition(h_t1_pred=h_t1_pred, h_t1_gt=h_t1_gt)
        if h_t.shape != h_t1_pred.shape:
            raise ValueError(
                "[LatentWorldPolicyBackend] cannot replace future DINO features with the current frame: "
                f"h_t={tuple(h_t.shape)}, h_t1_pred={tuple(h_t1_pred.shape)}."
            )
        # Same stop-grad contract as detach_future_feature. h_t is already a DINO feature.
        if self.training and bool(self.model_cfg.detach_future_feature):
            return h_t.detach()
        return h_t

    # ------------------
    # Checkpoint IO
    # ------------------
    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        save_dir = Path(str(save_directory))
        save_dir.mkdir(parents=True, exist_ok=True)

        out = self.vlm.save_pretrained(str(save_dir), **kwargs)
        ckpt_path = save_dir / "pytorch_model.pt"
        state_dict = self.state_dict()
        torch.save(state_dict, ckpt_path)
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[LatentWorldPolicyBackend] Full checkpoint saved to: {ckpt_path} (num_params={num_params})")
        return out

    # ------------------
    # Forward
    # ------------------
    def _prepare_queries(
        self, *, device: torch.device, vlm_stage_dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        flow_query = getattr(self, "flow_action_query", None)
        if flow_query is None:
            raise ValueError("[LatentWorldPolicyBackend] flow_action_query is None; check policy backend initialization.")
        vlm_embed_dtype = _module_param_dtype(self.vlm.get_input_embeddings(), default=vlm_stage_dtype)
        act_query = None if self.act_query is None else self.act_query.to(device=device, dtype=vlm_embed_dtype)
        flow_query = flow_query.to(device=device, dtype=vlm_embed_dtype)
        return act_query, flow_query

    def _prepare_train_batch(self, *, batch: LatentWorldPolicyTrainBatch) -> LatentWorldPolicyTrainBatch:
        return batch

    def _prepare_infer_batch(self, *, batch: LatentWorldPolicyInferBatch) -> LatentWorldPolicyInferBatch:
        return batch

    def _compute_fingertip_loss(
        self,
        *,
        h_t1_pred: torch.Tensor,
        pred_action_emb: torch.Tensor,
        state: torch.Tensor,
        fingertip_targets: torch.Tensor,
        fingertip_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fingertip_head is None:
            zero = h_t1_pred.new_zeros((), dtype=torch.float32)
            return zero, zero

        expected_points = int(self.model_cfg.fingertip_num_points)
        if self._fingertip_target_mode == "latent_delta":
            # Stage 2's predicted latent, not the frozen IDM code and not h_t1_pred.
            latent = pred_action_emb
            if latent.ndim == 2:
                latent = latent.unsqueeze(1)
            if latent.ndim != 3:
                raise ValueError(
                    "Expected predicted latent [B,Q,D] for latent_delta, "
                    f"got {tuple(pred_action_emb.shape)}."
                )
            if fingertip_targets.ndim != 3 or fingertip_targets.shape[1:] != (expected_points, 3):
                raise ValueError(
                    "Invalid latent_delta fingertip_targets shape: "
                    f"got={tuple(fingertip_targets.shape)}, expected=[B,{expected_points},3]."
                )
            if fingertip_valid.shape != fingertip_targets.shape[:2]:
                raise ValueError(
                    "Invalid latent_delta fingertip_valid shape: "
                    f"got={tuple(fingertip_valid.shape)}, expected={tuple(fingertip_targets.shape[:2])}."
                )
            scale = float(self.model_cfg.fingertip_target_scale)
            z_summary = latent.float().mean(dim=1)
            pred = self.fingertip_head(z_summary).view(z_summary.shape[0], expected_points, 3)
            target_meter = fingertip_targets.float()
            target = target_meter / scale
            pred_meter = pred * scale
            valid = fingertip_valid.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
            error = F.smooth_l1_loss(pred, target, reduction="none")
            denom = valid.sum().clamp_min(1.0)
            loss = (error * valid).sum() / (denom * 3.0)
            squared = ((pred_meter - target_meter) ** 2) * valid
            rmse = torch.sqrt(squared.sum() / (denom * 3.0)).detach()
            return loss, rmse

        if h_t1_pred.ndim != 3:
            raise ValueError(
                "Expected rolled-out future DINO tokens [B,K,D], "
                f"got {tuple(h_t1_pred.shape)}."
            )
        # This tensor stays in the graph. The flow head reads the same rollout.
        h_summary = h_t1_pred.float().mean(dim=1)
        if self._fingertip_target_mode == "endpoint":
            if fingertip_targets.ndim != 3 or fingertip_targets.shape[1:] != (expected_points, 3):
                raise ValueError(
                    "Invalid endpoint fingertip_targets shape: "
                    f"got={tuple(fingertip_targets.shape)}, expected=[B,{expected_points},3]."
                )
            if fingertip_valid.shape != fingertip_targets.shape[:2]:
                raise ValueError(
                    "Invalid endpoint fingertip_valid shape: "
                    f"got={tuple(fingertip_valid.shape)}, expected={tuple(fingertip_targets.shape[:2])}."
                )
            center = self._fingertip_endpoint_center(h_summary)
            scale = float(self.model_cfg.fingertip_target_scale)
            pred = self.fingertip_head(h_summary).view(h_summary.shape[0], expected_points, 3)
            target = (fingertip_targets.float() - center) / scale
            pred_meter = pred * scale + center
            target_meter = fingertip_targets.float()
        else:
            expected_horizon = int(self.model_cfg.action_horizon)
            if fingertip_targets.ndim != 4 or fingertip_targets.shape[1:] != (expected_horizon, expected_points, 3):
                raise ValueError(
                    "Invalid fingertip_targets shape: "
                    f"got={tuple(fingertip_targets.shape)}, expected=[B,{expected_horizon},{expected_points},3]."
                )
            if fingertip_valid.shape != fingertip_targets.shape[:3]:
                raise ValueError(
                    "Invalid fingertip_valid shape: "
                    f"got={tuple(fingertip_valid.shape)}, expected={tuple(fingertip_targets.shape[:3])}."
                )
            pred = self.fingertip_head(torch.cat([h_summary, state.float()], dim=-1))
            pred = pred.view(pred.shape[0], expected_horizon, expected_points, 3)
            target = fingertip_targets.float()
            pred_meter = pred
            target_meter = target
        valid = fingertip_valid.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)

        error = F.smooth_l1_loss(pred, target, reduction="none")
        denom = valid.sum().clamp_min(1.0)
        loss = (error * valid).sum() / (denom * 3.0)
        squared = ((pred_meter - target_meter) ** 2) * valid
        rmse = torch.sqrt(squared.sum() / (denom * 3.0)).detach()
        return loss, rmse

    def _fingertip_endpoint_center(self, like: torch.Tensor) -> torch.Tensor:
        raw = self.model_cfg.fingertip_target_center
        values = [float(value) for value in raw]
        if len(values) != 3:
            raise ValueError(
                "fingertip_target_center must contain 3 ego-camera coordinates in meters, "
                f"got {raw!r}."
            )
        return like.new_tensor(values).view(1, 1, 3)

    def _run_shared_encoding_core(
        self,
        *,
        prepared_batch: Union[LatentWorldPolicyInferBatch, LatentWorldPolicyTrainBatch],
        primary_visual_input: torch.Tensor,
        source: str,
        lam_features_with_no_grad: bool,
    ) -> PolicyEncodingState:
        vlm_stage_dtype = self.model_cfg.vlm_dtype
        lam_stage_dtype = torch.bfloat16

        device = prepared_batch["input_ids"].device
        act_query, flow_query = self._prepare_queries(device=device, vlm_stage_dtype=vlm_stage_dtype)

        with _cuda_autocast(vlm_stage_dtype):
            plan_pose_tokens = None
            if self.plan_pose_encoder is not None:
                pose = prepared_batch.get("object_pose_condition")
                if pose is None:
                    raise KeyError("Planning pose requires current object_pose_condition at train and inference time")
                plan_pose_tokens = self.plan_pose_encoder(pose)
            vlm_out_dict = self._run_vlm_stage(
                input_ids=prepared_batch["input_ids"],
                attention_mask=prepared_batch["attention_mask"],
                pixel_values=prepared_batch["pixel_values"],
                image_grid_thw=prepared_batch["image_grid_thw"],
                act_placeholder_mask=prepared_batch["act_placeholder_mask"],
                flow_placeholder_mask=prepared_batch["flow_placeholder_mask"],
                act_query=act_query,
                flow_query=flow_query,
                depth_placeholder_mask=prepared_batch.get("depth_placeholder_mask"),
                plan_pose_tokens=plan_pose_tokens,
            )
        h_vlm = vlm_out_dict["h_vlm"]
        pred_action_emb = vlm_out_dict["pred_latent"]

        with _cuda_autocast(lam_stage_dtype):
            if lam_features_with_no_grad:
                with torch.no_grad():
                    features = self.lam.extract_vision_features(primary_visual_input)
            else:
                features = self.lam.extract_vision_features(primary_visual_input)

            if features is None:
                raise ValueError(f"[{source}] lam visual feature extraction returned None; check LAM config.")
            h_t_original = features[:, 0, :, :]
            h_t = h_t_original
            h_t1_gt = features[:, -1, :, :]

            if self.model_cfg.future_prediction:
                h_t1_pred = self._decode_future_tokens_strict_single_query(
                    h_t=h_t,
                    pred_action_emb=pred_action_emb,
                    source=source,
                )
            else:
                h_t1_pred = h_t

        interaction_tokens, interaction_prediction = None, None
        if self.interaction_encoder is not None:
            with _cuda_autocast(torch.float32):
                interaction_tokens, interaction_prediction = self.interaction_encoder(
                    current_vision=h_t_original,vlm=h_vlm,
                    language_mask=prepared_batch["attention_mask"].bool(),
                    state=prepared_batch["state"],state_mask=prepared_batch["state_mask"],
                    geometry=prepared_batch.get("interaction_geometry"),
                    geometry_mask=prepared_batch.get("interaction_valid"))
        return PolicyEncodingState(
            h_vlm=h_vlm,
            pred_action_emb=pred_action_emb,
            h_t=h_t,
            h_t1_pred=h_t1_pred,
            h_t1_gt=h_t1_gt,
            h_t_original=h_t_original,
            interaction_tokens=interaction_tokens,interaction_prediction=interaction_prediction,
        )

    def _run_shared_encoding_train(
        self,
        *,
        prepared_batch: LatentWorldPolicyTrainBatch,
        source: str,
        lam_features_with_no_grad: bool,
    ) -> PolicyEncodingState:
        return self._run_shared_encoding_core(
            prepared_batch=prepared_batch,
            primary_visual_input=prepared_batch["primary_video"],
            source=source,
            lam_features_with_no_grad=lam_features_with_no_grad,
        )

    def _run_shared_encoding_infer(
        self,
        *,
        prepared_batch: LatentWorldPolicyInferBatch,
        source: str,
        lam_features_with_no_grad: bool,
    ) -> PolicyEncodingState:
        return self._run_shared_encoding_core(
            prepared_batch=prepared_batch,
            primary_visual_input=prepared_batch["primary_image"],
            source=source,
            lam_features_with_no_grad=lam_features_with_no_grad,
        )

    def forward(
        self,
        *,
        batch: LatentWorldPolicyTrainBatch,
    ) -> Dict[str, torch.Tensor]:
        # Precision contract (QwenGR00T-style):
        # - VLM stage: bf16 autocast
        # - LAM stage: bf16 autocast
        # - Flow/loss stage: float32 autocast
        lam_stage_dtype = torch.bfloat16
        flow_stage_dtype = torch.float32
        prepared_batch = cast(
            LatentWorldPolicyTrainBatch,
            self._prepare_train_batch(batch=batch),
        )
        device = prepared_batch["input_ids"].device

        shared = self._run_shared_encoding_train(
            prepared_batch=prepared_batch,
            source="LatentWorldPolicyBackend.forward",
            lam_features_with_no_grad=True,
        )

        with _cuda_autocast(lam_stage_dtype):
            loss_distill = torch.tensor(0.0, device=device, dtype=lam_stage_dtype)
            if bool(self.model_cfg.enable_loss_distill) and not self.model_cfg.action_expert_only:
                loss_distill = self._compute_distill_loss(
                    pred_latent=shared.pred_action_emb,
                    primary_video=prepared_batch["primary_video"],
                    embodiment_id=prepared_batch["embodiment_id"],
                )

            if self.model_cfg.future_prediction:
                loss_perceptual = F.mse_loss(shared.h_t1_pred, shared.h_t1_gt)
            else:
                loss_perceptual = torch.tensor(0.0, device=device, dtype=lam_stage_dtype)

        if float(self.model_cfg.fingertip_loss_weight) > 0.0 and not self.model_cfg.action_expert_only:
            fingertip_targets = prepared_batch.get("fingertip_targets")
            fingertip_valid = prepared_batch.get("fingertip_valid")
            if fingertip_targets is None or fingertip_valid is None:
                raise KeyError(
                    "Fingertip loss is enabled but the training batch has no fingertip_targets/fingertip_valid. "
                    "Configure datasets.vla_data.fingertip_sidecar_root and generate sidecars first."
                )
            loss_fingertip, fingertip_rmse = self._compute_fingertip_loss(
                h_t1_pred=shared.h_t1_pred,
                pred_action_emb=shared.pred_action_emb,
                state=prepared_batch["state"],
                fingertip_targets=fingertip_targets,
                fingertip_valid=fingertip_valid,
            )
        else:
            loss_fingertip = torch.zeros((), device=device, dtype=torch.float32)
            fingertip_rmse = torch.zeros((), device=device, dtype=torch.float32)
        fingertip_rmse_mm = fingertip_rmse * 1000.0

        loss_depth = torch.zeros((), device=device, dtype=torch.float32)
        if self.depth_align_head is not None and float(self.model_cfg.depth_loss_weight) > 0.0:
            depth_targets = prepared_batch.get("depth_targets")
            depth_mask = prepared_batch.get("depth_placeholder_mask")
            if depth_targets is None or depth_mask is None:
                raise KeyError(
                    "Depth loss is enabled but the batch has no depth_targets/depth_placeholder_mask."
                )
            loss_depth = self._compute_depth_loss(
                h_vlm=shared.h_vlm,
                input_ids=prepared_batch["input_ids"],
                image_grid_thw=prepared_batch["image_grid_thw"],
                depth_mask=depth_mask,
                depth_targets=depth_targets,
            )

        loss_wm_geometry = torch.zeros((), device=device, dtype=torch.float32)
        wm_position_rmse = torch.zeros_like(loss_wm_geometry)
        wm_angle_mae = torch.zeros_like(loss_wm_geometry)
        if self.wm_geometry_head is not None:
            if 'future_interaction_geometry' not in prepared_batch or 'future_interaction_valid' not in prepared_batch:
                raise KeyError('Future geometry supervision requires aligned future sidecar labels')
            with _cuda_autocast(torch.float32):
                prediction = self.wm_geometry_head(shared.h_t1_pred)
                loss_wm_geometry, wm_position_rmse, wm_angle_mae = self.wm_geometry_head.loss(
                    prediction, prepared_batch['future_interaction_geometry'], prepared_batch['future_interaction_valid'])

        loss_interaction = torch.zeros((),device=device,dtype=torch.float32)
        if self.interaction_encoder is not None:
            if self.model_cfg.interaction_mode == "visual_aux":
                if "interaction_geometry" not in prepared_batch or "interaction_valid" not in prepared_batch:
                    raise KeyError("visual_aux training requires audited current geometry targets")
                loss_interaction = self.interaction_encoder.auxiliary_loss(
                    shared.interaction_prediction,prepared_batch["interaction_geometry"],
                    prepared_batch["interaction_valid"])
            else:
                # Keep control readout parameters in the DDP graph without supervising geometry.
                loss_interaction = shared.interaction_prediction.sum()*0.0

        h_vlm_for_flow = _apply_flow_only_grad_to_h_vlm(
            h_vlm=shared.h_vlm,
            act_placeholder_mask=prepared_batch["act_placeholder_mask"],
            enable_flow_only=bool(self.model_cfg.flow_only_mode),
        )
        attn_flow = prepared_batch["attention_mask"] == 1

        with _cuda_autocast(flow_stage_dtype):
            repeat_steps = int(self.model_cfg.repeated_diffusion_steps)
            h_t_rep = shared.h_t.repeat(repeat_steps, *([1] * (shared.h_t.ndim - 1)))
            h_t1_pred_rep = shared.h_t1_pred.repeat(repeat_steps, *([1] * (shared.h_t1_pred.ndim - 1)))
            h_t1_gt_rep = shared.h_t1_gt.repeat(repeat_steps, *([1] * (shared.h_t1_gt.ndim - 1)))
            h_t1_cond_rep = self._select_flow_future_feature(
                h_t=h_t_rep,
                h_t1_pred=h_t1_pred_rep,
                h_t1_gt=h_t1_gt_rep,
            )
            h_vlm_rep = h_vlm_for_flow.repeat(repeat_steps, *([1] * (h_vlm_for_flow.ndim - 1)))
            state_rep = prepared_batch["state"].repeat(repeat_steps, *([1] * (prepared_batch["state"].ndim - 1)))
            state_mask_rep = prepared_batch["state_mask"].repeat(
                repeat_steps, *([1] * (prepared_batch["state_mask"].ndim - 1))
            )
            actions_rep = prepared_batch["actions"].repeat(
                repeat_steps, *([1] * (prepared_batch["actions"].ndim - 1))
            )
            actions_mask_rep = prepared_batch["actions_mask"].repeat(
                repeat_steps, *([1] * (prepared_batch["actions_mask"].ndim - 1))
            )
            action_hz_rep = prepared_batch["action_hz"].repeat(repeat_steps)
            embodiment_rep = prepared_batch["embodiment_id"].repeat(repeat_steps)
            attn_rep = attn_flow.repeat(repeat_steps, *([1] * (attn_flow.ndim - 1)))
            contact = prepared_batch.get("contact_geometry")
            contact_rep = None if contact is None else contact.repeat(repeat_steps, 1, 1)
            pose = prepared_batch.get("object_pose_condition")
            pose_rep = None if pose is None else pose.repeat(repeat_steps, 1)

            loss_flow = self.flow(
                h_t=h_t_rep,
                h_t1_star=h_t1_cond_rep,
                h_vlm=h_vlm_rep,
                state=state_rep,
                actions=actions_rep,
                action_hz=action_hz_rep,
                embodiment_id=embodiment_rep,
                state_mask=state_mask_rep,
                actions_mask=actions_mask_rep,
                attention_mask=attn_rep,
                contact_geometry=contact_rep,
                object_pose_condition=pose_rep,
                **({"interaction_tokens": shared.interaction_tokens.repeat(repeat_steps,1,1)}
                   if shared.interaction_tokens is not None else {}),
            )

            loss_total = (
                loss_flow
                + self.model_cfg.perceptual_weight * loss_perceptual
                + self.model_cfg.lam_encoder_distill_weight * loss_distill
                + self.model_cfg.fingertip_loss_weight * loss_fingertip
                + self.model_cfg.depth_loss_weight * loss_depth
                + self.model_cfg.interaction_loss_weight * loss_wm_geometry
                + (self.model_cfg.interaction_loss_weight if self.model_cfg.interaction_mode == "visual_aux" else 1.0) * loss_interaction
            )

        if self.model_cfg.action_expert_only:
            loss_total = loss_flow
        zero = torch.tensor(0.0, device=device, dtype=loss_total.dtype)
        return {
            "loss_flow": loss_flow,
            "loss_perceptual": loss_perceptual,
            "loss_distill": loss_distill,
            "loss_vlm": zero,
            "loss_fingertip": loss_fingertip,
            "fingertip_rmse": fingertip_rmse,
            "fingertip_rmse_mm": fingertip_rmse_mm,
            "loss_depth": loss_depth,
            **({"loss_wm_geometry": loss_wm_geometry, "wm_position_rmse_m": wm_position_rmse,
                "wm_angle_mae_rad": wm_angle_mae} if self.wm_geometry_head is not None else {}),
            **({"loss_interaction":loss_interaction} if self.interaction_encoder is not None else {}),
            "loss_total": loss_total,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        *,
        batch: LatentWorldPolicyInferBatch,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        return_intermediates: bool = False,
        return_padded: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        flow_stage_dtype = torch.float32
        prepared_batch = cast(
            LatentWorldPolicyInferBatch,
            self._prepare_infer_batch(batch=batch),
        )

        if guidance_scale is None:
            guidance_scale = float(self.flow.config.cfg_guidance_scale)
        if num_inference_steps is None:
            num_inference_steps = int(self.flow.config.num_inference_steps)

        shared = self._run_shared_encoding_infer(
            prepared_batch=prepared_batch,
            source="LatentWorldPolicyBackend.predict_action",
            lam_features_with_no_grad=False,
        )
        attn_flow = prepared_batch["attention_mask"] == 1

        with _cuda_autocast(flow_stage_dtype):
            if str(self.model_cfg.flow_future_feature).lower() == "current":
                h_t1_for_flow = shared.h_t
            else:
                h_t1_for_flow = shared.h_t1_pred
            actions = self.flow.sample_actions_cfg(
                h_t=shared.h_t,
                h_t1_star=h_t1_for_flow,
                h_vlm=shared.h_vlm,
                state=prepared_batch["state"],
                state_mask=prepared_batch["state_mask"],
                action_hz=prepared_batch["action_hz"],
                embodiment_id=prepared_batch["embodiment_id"],
                cfg_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                attention_mask=attn_flow,
                return_padded=bool(return_padded),
                contact_geometry=prepared_batch.get("contact_geometry"),
                object_pose_condition=prepared_batch.get("object_pose_condition"),
                **({"interaction_tokens":shared.interaction_tokens} if shared.interaction_tokens is not None else {}),
            )

        if not return_intermediates:
            return actions

        num_tokens = shared.h_t_original.shape[1]
        hw = int(num_tokens ** 0.5)
        if hw * hw != num_tokens:
            hw = 16
        intermediates = {
            "h_t": shared.h_t_original.detach().cpu(),
            "h_t1_pred": shared.h_t1_pred.detach().cpu(),
            "vision_tokens_hw": (hw, hw),
        }
        return actions, intermediates
