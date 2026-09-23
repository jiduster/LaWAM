#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"h5py is required: {exc}") from exc


def _json_attr(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return json.loads(value)
    except Exception:
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/ceph3/dexwm/robocasa_random_data"))
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    roots = [
        args.root / "exploratory_movements",
        args.root / "exploratory_movement",
        args.root / "gripper_open_and_close",
        args.root / "pick-and-place-2.0",
    ]

    count = 0
    for root in roots:
        if not root.exists():
            continue
        for h5_path in sorted(root.glob("*.hdf5")):
            with h5py.File(h5_path, "r") as f:
                if "data" not in f:
                    continue
                env_args = _json_attr(f["data"].attrs.get("env_args", "{}"))
                for demo_name in sorted(f["data"].keys()):
                    demo = f["data"][demo_name]
                    obs = demo["obs"]
                    action_key = "abs_right_arm_base_action" if "abs_right_arm_base_action" in demo else "actions"
                    summary = {
                        "file": str(h5_path),
                        "demo": demo_name,
                        "subdir": root.name,
                        "length": int(demo[action_key].shape[0]),
                        "action_key": action_key,
                        "action_shape": list(map(int, demo[action_key].shape)),
                        "state_shape": list(map(int, demo["states"].shape)) if "states" in demo else None,
                        "env_name": env_args.get("env_name") if isinstance(env_args, dict) else None,
                        "robots": env_args.get("env_kwargs", {}).get("robots") if isinstance(env_args, dict) else None,
                        "lang": _json_attr(demo.attrs.get("ep_meta", "{}")).get("free_form_lang")
                        if isinstance(_json_attr(demo.attrs.get("ep_meta", "{}")), dict)
                        else None,
                        "has_primary": "robot0_robotview_2_image" in obs,
                        "has_wrist": "gripper0_right_right_eye_in_hand_image" in obs,
                    }
                    print(json.dumps(summary, ensure_ascii=False))
                    count += 1
                    if count >= int(args.limit):
                        return


if __name__ == "__main__":
    main()
