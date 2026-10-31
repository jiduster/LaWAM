from __future__ import annotations

from typing import Any, Sequence

import torch
from starVLA.contact_geometry import ContactStore
from starVLA.model.framework.vlas.pose_conditioning import pose_conditioning_enabled

from starVLA.model.framework.latent_world.batch_utils import (
    align_feature_dim_with_mask,
    apply_shared_color_jitter_uint8,
    apply_shared_random_resized_crop_uint8,
    build_placeholder_masks,
    imagenet_normalize_video_,
    sample_or_pad_sequence_with_mask,
)
from starVLA.model.framework.latent_world.processor_utils import (
    LatentWorldProcessorSpec,
    load_latent_world_processor,
)
from starVLA.model.framework.latent_world.types import LatentWorldPolicyTrainBatch, LatentWorldPolicyTrainRawSample
from starVLA.model.framework.vlas.depth_queries import DepthTargetStore
from starVLA.model.framework.latent_world.vlm_adapter import build_qwenvl_messages
from starVLA.model.framework.latent_world.vlm_adapter import (
    DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT,
    DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT,
)


class LatentWorldTrainCollator:
    def __init__(
        self,
        *,
        policy_cfg: Any,
        processor_spec: LatentWorldProcessorSpec,
        act_queries: int,
        flow_queries: int,
        enable_primary_video_aug: bool,
        enable_primary_random_resized_crop: bool,
        cot_prompt_before_wrist: str | None = None,
        cot_prompt_after_wrist: str | None = None,
    ) -> None:
        self.policy_cfg = policy_cfg
        self.processor_spec = processor_spec
        self.placeholder_token = str(processor_spec.placeholder_token)
        self.act_queries = int(act_queries)
        self.flow_queries = int(flow_queries)
        self.enable_primary_video_aug = bool(enable_primary_video_aug)
        self.enable_primary_random_resized_crop = bool(enable_primary_random_resized_crop)
        self.cot_prompt_before_wrist = (
            str(cot_prompt_before_wrist)
            if cot_prompt_before_wrist is not None
            else DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT
        )
        self.cot_prompt_after_wrist = (
            str(cot_prompt_after_wrist)
            if cot_prompt_after_wrist is not None
            else DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT
        )
        self.training = True
        self.pose_store = None
        if pose_conditioning_enabled(policy_cfg):
            from starVLA.object_pose import ObjectPoseStore
            self.pose_store = ObjectPoseStore(policy_cfg.object_pose_sidecar_root)
        self.contact_store = None
        if bool(getattr(policy_cfg.flow_cfg, "contact_conditioning", False)):
            self.contact_store = ContactStore(policy_cfg.contact_sidecar_root)
        self.depth_enabled = bool(getattr(policy_cfg, "depth_query_conditioning", False))
        self.depth_queries_per_view = int(getattr(policy_cfg, "depth_num_task_tokens", 8)) if self.depth_enabled else 0
        self.depth_num_views = int(getattr(policy_cfg, "depth_num_views", 3))
        self.depth_placeholder_token = str(getattr(policy_cfg, "depth_placeholder_token", "<DEPTH_PH>"))
        self.depth_store = None
        if self.depth_enabled:
            self.depth_store = DepthTargetStore(getattr(policy_cfg, "depth_feature_root", ""))
        self._processor: Any | None = None
        self._placeholder_token_id: int | None = None
        self._depth_placeholder_token_id: int | None = None

    def train(self) -> "LatentWorldTrainCollator":
        self.training = True
        return self

    def eval(self) -> "LatentWorldTrainCollator":
        self.training = False
        return self

    def _ensure_processor(self) -> tuple[Any, int, int | None]:
        if self._processor is None:
            processor, _, placeholder_token_id, depth_token_id = load_latent_world_processor(self.processor_spec)
            self._processor = processor
            self._placeholder_token_id = placeholder_token_id
            self._depth_placeholder_token_id = depth_token_id
        assert self._processor is not None
        assert self._placeholder_token_id is not None
        if self.depth_enabled and self._depth_placeholder_token_id is None:
            raise RuntimeError("Depth query conditioning is enabled but the depth placeholder token was not registered.")
        return self._processor, self._placeholder_token_id, self._depth_placeholder_token_id

    @staticmethod
    def _to_wrist_view_list(wrist_images: torch.Tensor) -> list[torch.Tensor]:
        # Match inference semantics: missing wrist views are represented as an empty view list.
        if int(wrist_images.shape[0]) == 0:
            return []
        return [img.contiguous() for img in wrist_images]

    @staticmethod
    def _to_primary_view_list(primary_videos: torch.Tensor) -> list[torch.Tensor]:
        if primary_videos.ndim != 5 or int(primary_videos.shape[1]) < 1:
            raise ValueError(
                "Expected primary_videos with shape [V, T, C, H, W] and T>=1, "
                f"got {tuple(primary_videos.shape)}."
            )
        return [video[0].contiguous() for video in primary_videos]

    @staticmethod
    def _augment_primary_and_wrist_views(
        *,
        primary_videos: torch.Tensor,
        wrist_images: torch.Tensor,
        enable_random_resized_crop: bool,
        enable_color_jitter: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if primary_videos.ndim != 5:
            raise ValueError(f"Expected primary_videos with shape [V, T, C, H, W], got {tuple(primary_videos.shape)}.")
        primary_view_count = int(primary_videos.shape[0])
        primary_frame_count = int(primary_videos.shape[1])
        flat_primary = primary_videos.reshape(primary_view_count * primary_frame_count, *primary_videos.shape[-3:])

        stacked_views = [flat_primary]
        if int(wrist_images.shape[0]) > 0:
            stacked_views.append(wrist_images)
        augmented_views = torch.cat(stacked_views, dim=0)
        if enable_random_resized_crop:
            augmented_views = apply_shared_random_resized_crop_uint8(augmented_views)
        if enable_color_jitter:
            augmented_views = apply_shared_color_jitter_uint8(augmented_views)

        primary_flat_count = int(flat_primary.shape[0])
        processed_primary_videos = augmented_views[:primary_flat_count].reshape_as(primary_videos).contiguous()
        processed_wrist_images = augmented_views[primary_flat_count:].contiguous()
        return processed_primary_videos, processed_wrist_images

    def __call__(self, features: Sequence[LatentWorldPolicyTrainRawSample]) -> LatentWorldPolicyTrainBatch:
        if len(features) == 0:
            raise ValueError("LatentWorldTrainCollator received an empty feature list.")

        processor, placeholder_token_id, depth_token_id = self._ensure_processor()

        image_views_batch: list[list[Any]] = []
        wrist_image_views_batch: list[list[Any]] = []
        primary_video_uint8_seqs: list[torch.Tensor] = []
        depth_target_seqs: list[torch.Tensor] = []
        contact_frames: list[torch.Tensor] = []
        pose_frames: list[torch.Tensor] = []
        interaction_frames, interaction_masks = [], []
        future_geometry, future_masks = [], []
        from starVLA.model.framework.vlas.wm_geometry import FUTURE_GEOMETRY_MODES
        wm_labels = getattr(self.policy_cfg, 'interaction_mode', 'none') in FUTURE_GEOMETRY_MODES
        interaction_labels = getattr(self.policy_cfg, "interaction_mode", "none") in {
            "oracle", "visual_aux", "visual_joint_aux"}
        pose_enabled = pose_conditioning_enabled(self.policy_cfg)
        instructions: list[str] = []
        actions_list: list[torch.Tensor] = []
        states_list: list[torch.Tensor] = []
        embodiment_ids: list[int] = []
        action_hz_list: list[float] = []
        fingertip_targets_list: list[torch.Tensor] = []
        fingertip_valid_list: list[torch.Tensor] = []
        fingertip_enabled = (float(getattr(self.policy_cfg, "fingertip_loss_weight", 0.0)) > 0.0
                             and not bool(getattr(self.policy_cfg, "action_expert_only", False)))
        window_size = int(self.policy_cfg.action_horizon)

        for sample in features:
            primary_videos = sample["primary_videos"].contiguous()
            wm_primary_video = sample.get("wm_primary_video", primary_videos[0]).contiguous()
            wrist_images = sample["wrist_images"].contiguous()
            state_tensor = sample["state"].to(dtype=torch.float32).contiguous()
            action_tensor = sample["action"].to(dtype=torch.float32).contiguous()

            instruction = str(sample["lang"])
            embodiment_id = int(sample["embodiment_id"])
            action_hz = float(sample["action_hz"])
            if wm_labels:
                from starVLA.interaction_geometry import validate_geometry
                if 'future_interaction_geometry' not in sample or 'future_interaction_valid' not in sample:
                    raise KeyError('WM geometry requires future_interaction_geometry/future_interaction_valid')
                value = torch.as_tensor(sample['future_interaction_geometry'], dtype=torch.float32)
                valid = torch.as_tensor(sample['future_interaction_valid'], dtype=torch.bool)
                expected = (int(self.policy_cfg.interaction_entities), int(self.policy_cfg.interaction_feature_dim))
                if tuple(value.shape) != expected or valid.shape != value.shape:
                    raise ValueError('Future geometry sample shape mismatch')
                if valid.any():
                    validate_geometry(value.numpy(), valid.numpy(), self.policy_cfg.interaction_task)
                future_geometry.append(value)
                future_masks.append(valid)
            if interaction_labels:
                from starVLA.interaction_geometry import validate_geometry
                if "interaction_geometry" not in sample or "interaction_valid" not in sample:
                    raise KeyError("oracle/visual_aux/visual_joint_aux requires current interaction_geometry/interaction_valid sidecar labels")
                value = torch.as_tensor(sample["interaction_geometry"], dtype=torch.float32)
                valid = torch.as_tensor(sample["interaction_valid"], dtype=torch.bool)
                allow_empty = (
                    getattr(self.policy_cfg, "interaction_mode", "none") != "oracle"
                    and bool(getattr(self.policy_cfg, "interaction_allow_empty_frames", False))
                )
                validate_geometry(
                    value.numpy(), valid.numpy(), self.policy_cfg.interaction_task,
                    allow_empty_frames=allow_empty)
                interaction_frames.append(value)
                interaction_masks.append(valid)
            if pose_enabled:
                from starVLA.model.framework.vlas.pose_conditioning import validate_pose_tensor
                value = sample.get("object_pose_condition")
                if value is None:
                    value = self.pose_store.frame(sample["dataset_name"], int(sample["episode_index"]),
                                                  int(sample["frame_index"]), action_hz)
                value = torch.as_tensor(value, dtype=torch.float32)
                if value.shape != (12,):
                    raise ValueError("Training sample object_pose_condition must be [12]")
                validate_pose_tensor(value.unsqueeze(0))
                pose_frames.append(value)
            if self.contact_store is not None:
                contact_frames.append(torch.from_numpy(self.contact_store.frame(
                    sample["dataset_name"], int(sample["episode_index"]),
                    int(sample["frame_index"]), action_hz,
                )))

            fingertip_positions = sample.get("fingertip_positions")
            fingertip_valid = sample.get("fingertip_valid")
            if fingertip_enabled and (fingertip_positions is None or fingertip_valid is None):
                raise KeyError(
                    "Fingertip supervision is enabled but a dataset sample has no fingertip sidecar data. "
                    "Set datasets.vla_data.fingertip_sidecar_root and generate all episode sidecars."
                )
            if fingertip_positions is not None or fingertip_valid is not None:
                if fingertip_positions is None or fingertip_valid is None:
                    raise KeyError("Fingertip sample must contain both fingertip_positions and fingertip_valid.")
                fingertip_positions = torch.as_tensor(fingertip_positions, dtype=torch.float32).contiguous()
                fingertip_valid = torch.as_tensor(fingertip_valid, dtype=torch.bool).contiguous()
                if fingertip_positions.ndim != 3 or fingertip_positions.shape[-1] != 3:
                    raise ValueError(
                        "Expected fingertip_positions with shape [H,N,3], "
                        f"got {tuple(fingertip_positions.shape)}."
                    )
                if fingertip_valid.shape != fingertip_positions.shape[:2]:
                    raise ValueError(
                        "Expected fingertip_valid with shape [H,N], "
                        f"got {tuple(fingertip_valid.shape)} for positions {tuple(fingertip_positions.shape)}."
                    )
                if int(fingertip_positions.shape[0]) != int(window_size):
                    raise ValueError(
                        "Fingertip horizon does not match action_horizon: "
                        f"tips={int(fingertip_positions.shape[0])}, action_horizon={window_size}."
                    )
                target_mode = str(getattr(self.policy_cfg, "fingertip_target_mode", "delta")).lower()
                if target_mode == "endpoint":
                    # Last action-chunk frame, aligned with the future DINO token.
                    fingertip_targets_list.append(fingertip_positions[-1])
                    fingertip_valid_list.append(fingertip_valid[-1])
                elif target_mode == "latent_delta":
                    # Displacement across the two frames the Stage 2 latent represents.
                    fingertip_targets_list.append(fingertip_positions[-1] - fingertip_positions[0])
                    fingertip_valid_list.append(fingertip_valid[0] & fingertip_valid[-1])
                elif target_mode == "delta":
                    fingertip_targets_list.append(fingertip_positions - fingertip_positions[:1])
                    fingertip_valid_list.append(fingertip_valid)
                else:
                    raise ValueError(
                        "fingertip_target_mode must be 'delta', 'endpoint', or 'latent_delta', "
                        f"got {target_mode!r}."
                    )

            processed_primary_videos = primary_videos
            processed_wm_primary_video = wm_primary_video
            processed_wrist_images = wrist_images
            if self.training and (self.enable_primary_random_resized_crop or self.enable_primary_video_aug):
                # Apply the same spatial/color augmentation to VLM primary views,
                # the WM primary view, and wrist views so their geometry remains aligned.
                augmented_primary_videos, processed_wrist_images = self._augment_primary_and_wrist_views(
                    primary_videos=torch.cat([primary_videos, wm_primary_video.unsqueeze(0)], dim=0),
                    wrist_images=wrist_images,
                    enable_random_resized_crop=self.enable_primary_random_resized_crop,
                    enable_color_jitter=self.enable_primary_video_aug,
                )
                processed_primary_videos = augmented_primary_videos[:-1].contiguous()
                processed_wm_primary_video = augmented_primary_videos[-1].contiguous()

            primary_views = self._to_primary_view_list(processed_primary_videos)
            wrist_views = self._to_wrist_view_list(processed_wrist_images)
            image_views_batch.append(primary_views)
            wrist_image_views_batch.append(wrist_views)
            if self.depth_enabled:
                view_count = len(primary_views) + len(wrist_views)
                if view_count != self.depth_num_views:
                    raise ValueError(
                        "Depth queries expect "
                        f"{self.depth_num_views} camera views, got {view_count} "
                        f"({len(primary_views)} primary, {len(wrist_views)} wrist)."
                    )
                if "episode_index" not in sample or "frame_index" not in sample:
                    raise KeyError("Depth targets require episode_index and frame_index on each training sample.")
                if self.depth_store is None:
                    raise RuntimeError("Depth target store was not initialized.")
                depth_target_seqs.append(
                    self.depth_store.frame(int(sample["episode_index"]), int(sample["frame_index"]))
                )
            primary_video_uint8_seqs.append(processed_wm_primary_video.contiguous())
            instructions.append(instruction)
            actions_list.append(action_tensor)
            states_list.append(state_tensor)
            embodiment_ids.append(embodiment_id)
            action_hz_list.append(action_hz)

        qwen_messages = build_qwenvl_messages(
            images=image_views_batch,
            wrist_images=wrist_image_views_batch,
            instructions=instructions,
            placeholder_token=self.placeholder_token,
            act_queries=self.act_queries,
            flow_queries=self.flow_queries,
            cot_prompt_before_wrist=self.cot_prompt_before_wrist,
            cot_prompt_after_wrist=self.cot_prompt_after_wrist,
            depth_queries_per_view=self.depth_queries_per_view,
            depth_placeholder_token=self.depth_placeholder_token,
        )
        qwen_inputs = processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = qwen_inputs["input_ids"]
        act_mask, flow_mask = build_placeholder_masks(
            input_ids,
            act_queries=self.act_queries,
            flow_queries=self.flow_queries,
            placeholder_id=placeholder_token_id,
        )

        action_dim = int(self.policy_cfg.flow_cfg.action_dim)
        window_size = int(self.policy_cfg.action_horizon)
        state_dim = int(self.policy_cfg.flow_cfg.state_dim)

        action_tensors = []
        action_masks = []
        state_tensors = []
        state_masks = []
        for idx, state_tensor in enumerate(states_list):
            action_tensor = actions_list[idx]
            action_tensor, action_dim_mask = align_feature_dim_with_mask(action_tensor, action_dim)
            action_tensor, action_time_mask = sample_or_pad_sequence_with_mask(action_tensor, window_size)
            action_tensors.append(action_tensor)
            action_masks.append(action_time_mask.unsqueeze(1) & action_dim_mask.unsqueeze(0))

            aligned_state, state_dim_mask = align_feature_dim_with_mask(state_tensor, state_dim)
            state_tensors.append(aligned_state[-1].contiguous())
            state_masks.append(state_dim_mask.contiguous())

        primary_video_uint8_batch = torch.stack(primary_video_uint8_seqs, dim=0)
        primary_video_batch = primary_video_uint8_batch.to(dtype=torch.float32).div_(255.0)
        imagenet_normalize_video_(primary_video_batch)

        batch: LatentWorldPolicyTrainBatch = {
            "pixel_values": qwen_inputs["pixel_values"],
            "input_ids": input_ids,
            "attention_mask": qwen_inputs["attention_mask"],
            "act_placeholder_mask": act_mask,
            "flow_placeholder_mask": flow_mask,
            "primary_video": primary_video_batch,
            "state": torch.stack(state_tensors, dim=0),
            "state_mask": torch.stack(state_masks, dim=0),
            "embodiment_id": torch.tensor(embodiment_ids, dtype=torch.long),
            "action_hz": torch.tensor(action_hz_list, dtype=torch.float32),
            "image_grid_thw": qwen_inputs.get("image_grid_thw"),
            "actions": torch.stack(action_tensors, dim=0),
            "actions_mask": torch.stack(action_masks, dim=0),
        }
        if not torch.is_tensor(batch["image_grid_thw"]):
            batch["image_grid_thw"] = None
        if fingertip_enabled:
            batch["fingertip_targets"] = torch.stack(fingertip_targets_list, dim=0)
            batch["fingertip_valid"] = torch.stack(fingertip_valid_list, dim=0)
        if pose_enabled:
            batch["object_pose_condition"] = torch.stack(pose_frames)
        if interaction_labels:
            batch["interaction_geometry"] = torch.stack(interaction_frames)
            batch["interaction_valid"] = torch.stack(interaction_masks)
        if wm_labels:
            batch['future_interaction_geometry'] = torch.stack(future_geometry)
            batch['future_interaction_valid'] = torch.stack(future_masks)
        if self.contact_store is not None:
            batch["contact_geometry"] = torch.stack(contact_frames)
        if self.depth_enabled:
            if depth_token_id is None:
                raise RuntimeError("Depth placeholder token id is missing.")
            batch["depth_placeholder_mask"] = input_ids == int(depth_token_id)
            batch["depth_targets"] = torch.stack(depth_target_seqs, dim=0).contiguous()
        return batch
