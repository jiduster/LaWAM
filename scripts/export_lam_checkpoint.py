"""Export only the LAM module from a Lightning Stage 1 checkpoint.

The Stage 1 fingertip predictor is a training-only auxiliary head. Stage 2's
strict LAM loader expects a checkpoint containing only ``lam.*`` keys, so this
script removes the Lightning wrapper and excludes that auxiliary head.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export_lam_checkpoint(input_path: Path, output_path: Path) -> int:
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            f"Expected a Lightning checkpoint with a `state_dict`: {input_path}"
        )
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint state_dict must be a dict, got {type(state_dict).__name__}")

    lam_state = {}
    for key, value in state_dict.items():
        if key.startswith("module.lam."):
            lam_state["lam." + key[len("module.lam."):]] = value
        elif key.startswith("lam."):
            lam_state[key] = value
    if not lam_state:
        raise ValueError(f"No `lam.*` parameters found in checkpoint: {input_path}")

    unexpected = sorted(
        key
        for key in state_dict
        if key.startswith("fingertip_head.") or key.startswith("module.fingertip_head.")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": lam_state}, output_path)
    print(f"Exported {len(lam_state)} LAM tensors to {output_path}")
    if unexpected:
        print(f"Excluded {len(unexpected)} fingertip-head tensors")
    return len(lam_state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Lightning .ckpt")
    parser.add_argument("--output", type=Path, required=True, help="LAM .pt for Stage 2")
    args = parser.parse_args()
    export_lam_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
