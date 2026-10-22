from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch
from typing_extensions import NotRequired, Required, TypedDict

TensorLike2D = Union[torch.Tensor, np.ndarray, Sequence[float], Sequence[Sequence[float]]]
FrameArray = np.ndarray
ImageViews = Sequence[FrameArray]
VideoViews = Sequence[Sequence[FrameArray]]


class LatentWorldPolicyTrainRawSample(TypedDict):
    object_pose_condition: NotRequired[torch.Tensor]
    primary_videos: torch.Tensor
    wm_primary_video: NotRequired[torch.Tensor]
    wrist_images: torch.Tensor
    lang: str
    state: torch.Tensor
    action: torch.Tensor
    embodiment_id: int
    action_hz: float
    fingertip_positions: NotRequired[torch.Tensor]
    fingertip_valid: NotRequired[torch.Tensor]
    episode_index: NotRequired[int]
    frame_index: NotRequired[int]
    dataset_name: NotRequired[str]


class LatentWorldPolicyInferExample(TypedDict, total=False):
    primary_image: Required[ImageViews]
    wm_primary_image: NotRequired[FrameArray]
    lang: Required[str]
    embodiment_id: Required[int]
    action_hz: Required[float]
    state: NotRequired[TensorLike2D]
    state_mask: NotRequired[TensorLike2D]
    wrist_image: NotRequired[ImageViews]
    object_pose_condition: NotRequired[TensorLike2D]
    contact_geometry: NotRequired[TensorLike2D]


class LatentWorldPolicyInferBatch(TypedDict):
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    act_placeholder_mask: torch.Tensor
    flow_placeholder_mask: torch.Tensor
    primary_image: torch.Tensor
    object_pose_condition: NotRequired[torch.Tensor]
    contact_geometry: NotRequired[torch.Tensor]
    state: torch.Tensor
    state_mask: torch.Tensor
    embodiment_id: torch.Tensor
    action_hz: torch.Tensor
    image_grid_thw: Optional[torch.Tensor]
    depth_placeholder_mask: NotRequired[torch.Tensor]


class LatentWorldPolicyTrainBatch(TypedDict):
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    act_placeholder_mask: torch.Tensor
    flow_placeholder_mask: torch.Tensor
    primary_video: torch.Tensor
    object_pose_condition: NotRequired[torch.Tensor]
    contact_geometry: NotRequired[torch.Tensor]
    state: torch.Tensor
    state_mask: torch.Tensor
    embodiment_id: torch.Tensor
    action_hz: torch.Tensor
    image_grid_thw: Optional[torch.Tensor]
    actions: torch.Tensor
    actions_mask: torch.Tensor
    fingertip_targets: NotRequired[torch.Tensor]
    fingertip_valid: NotRequired[torch.Tensor]
    depth_placeholder_mask: NotRequired[torch.Tensor]
    depth_targets: NotRequired[torch.Tensor]
