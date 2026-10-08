from typing import Any, Dict, Sequence

import torch


def lam_collate(batch: Sequence[Dict], max_state_dim: int = 32) -> Dict[str, Any]:
    """
    Collate fixed-format clips and proprio into a batch.

    Input contract (upstream dataset must guarantee):
    - Each sample: {"frames": Tensor (T,C,H,W) uint8, "proprio": Tensor (T,D) float32, "embodiment_id": int}.
    - Within a batch: all samples have the same (T,C,H,W), same T for proprio.
      If D > max_state_dim, the proprio vector is truncated to the first max_state_dim dimensions.

    If fingertip sidecar fields are present in every sample, also returns:
        fingertip_delta: [B,N,3] endpoint displacement in meters
        fingertip_valid: [B,N] endpoint-valid mask

    Returns:
        videos: [B,T,C,H,W] uint8
        states: [B,2,max_state_dim] float32 (start/end proprio with right padding)
        state_mask: [B,2,max_state_dim] bool
        proprio: [B,2,max_state_dim] float32 (compat alias for states)
        delta_proprio: [B,max_state_dim] float32
        embodiment_ids: [B] int64
        proprio_mask: [B,max_state_dim] float32 (compat dim-only mask)
        dataset_names: list[str] (optional sample provenance)
        trajectory_ids: list[Union[int, str]] (optional sample provenance)
        base_indices: [B] int64 (optional sample provenance)
    """
    if len(batch) == 0:
        raise ValueError("lam_collate received an empty batch.")

    batch_size = len(batch)
    videos = torch.stack([s["frames"] for s in batch], dim=0)
    states_t = torch.zeros((batch_size, 2, max_state_dim), dtype=torch.float32)
    state_mask_t = torch.zeros((batch_size, 2, max_state_dim), dtype=torch.bool)
    delta_t = torch.zeros((batch_size, max_state_dim), dtype=torch.float32)
    proprio_mask_t = torch.zeros((batch_size, max_state_dim), dtype=torch.float32)
    has_fingertip = [
        "fingertip_positions" in sample or "fingertip_valid" in sample
        for sample in batch
    ]
    if any(has_fingertip) and not all(has_fingertip):
        raise ValueError("Fingertip sidecar data must be present for every sample in a batch.")
    fingertip_delta_t = None
    fingertip_valid_t = None
    if all(has_fingertip):
        first_positions = torch.as_tensor(batch[0]["fingertip_positions"])
        if first_positions.ndim != 3 or first_positions.shape[-1] != 3:
            raise ValueError(
                "Expected fingertip_positions with shape [T,N,3], "
                f"got {tuple(first_positions.shape)}."
            )
        num_points = int(first_positions.shape[1])
        fingertip_delta_t = torch.zeros((batch_size, num_points, 3), dtype=torch.float32)
        fingertip_valid_t = torch.zeros((batch_size, num_points), dtype=torch.bool)
    for i, sample in enumerate(batch):
        proprio = sample["proprio"]
        if not isinstance(proprio, torch.Tensor):
            proprio = torch.as_tensor(proprio)
        proprio = proprio.to(torch.float32)
        if proprio.ndim != 2:
            raise ValueError(
                f"Expected sample['proprio'] to have shape [T, D], got {tuple(proprio.shape)} "
                f"for sample index {i}."
            )
        D = int(proprio.shape[1])
        if D <= 0:
            raise ValueError(f"Invalid proprio dim D={D} for sample index {i}.")
        if D > max_state_dim:
            proprio = proprio[:, :max_state_dim]
            D = int(max_state_dim)

        start = proprio[0, :]
        end = proprio[-1, :]
        states_t[i, 0, :D] = start
        states_t[i, 1, :D] = end
        state_mask_t[i, :, :D] = True
        delta_t[i, :D] = end - start
        proprio_mask_t[i, :D] = 1.0

        if all(has_fingertip):
            positions = torch.as_tensor(sample["fingertip_positions"], dtype=torch.float32)
            valid = torch.as_tensor(sample["fingertip_valid"], dtype=torch.bool)
            if positions.shape != (positions.shape[0], num_points, 3):
                raise ValueError(
                    "All fingertip positions must have shape [T,N,3] with a fixed N. "
                    f"Sample {i} has {tuple(positions.shape)}."
                )
            if valid.shape != positions.shape[:2]:
                raise ValueError(
                    "Expected fingertip_valid shape [T,N], "
                    f"got {tuple(valid.shape)} for positions {tuple(positions.shape)}."
                )
            if positions.shape[0] < 2:
                raise ValueError("Stage 1 fingertip supervision requires at least two temporal endpoints.")
            fingertip_delta_t[i] = positions[-1] - positions[0]
            fingertip_valid_t[i] = valid[0] & valid[-1]

    embodiment_ids_t = torch.tensor(
        [s["embodiment_id"] for s in batch],
        dtype=torch.long,
    )
    dataset_names = [str(s.get("dataset_name", "unknown")) for s in batch]
    trajectory_ids = [s.get("trajectory_id", -1) for s in batch]
    base_indices = torch.tensor(
        [int(s.get("base_index", -1)) for s in batch],
        dtype=torch.long,
    )

    output = {
        "videos": videos,
        "states": states_t,
        "state_mask": state_mask_t,
        "proprio": states_t,
        "delta_proprio": delta_t,
        "embodiment_ids": embodiment_ids_t,
        "proprio_mask": proprio_mask_t,
        "dataset_names": dataset_names,
        "trajectory_ids": trajectory_ids,
        "base_indices": base_indices,
    }
    if fingertip_delta_t is not None:
        output["fingertip_delta"] = fingertip_delta_t
        output["fingertip_valid"] = fingertip_valid_t
    return output
