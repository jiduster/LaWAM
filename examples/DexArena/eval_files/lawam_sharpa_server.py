"""Websocket inference server for LaWAM checkpoints trained on Sharpa data.

The wire request is the repository websocket protocol with an ``examples`` list.  Each
example contains uint8 HWC images and raw controller-qpos; normalization is deliberately
kept on this side so a DexArena client never needs to read checkpoint statistics.
"""

from __future__ import annotations

import argparse
import logging
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework


def _min_max_normalize(value: Any, stats: dict[str, Any]) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    low = np.asarray(stats["min"], dtype=np.float32)
    high = np.asarray(stats["max"], dtype=np.float32)
    if x.shape[-1] != low.shape[0] or high.shape != low.shape:
        raise ValueError(f"state dimension {x.shape[-1]} != statistics dimension {low.shape[0]}")
    out = np.zeros_like(x, dtype=np.float32)
    mask = high != low
    out[..., mask] = 2.0 * (x[..., mask] - low[mask]) / (high[mask] - low[mask]) - 1.0
    # Match StateActionTransform's min_max behavior (constant dimensions become zero).
    return out


def _min_max_unnormalize(value: Any, stats: dict[str, Any]) -> np.ndarray:
    """Inverse of StateActionTransform min_max: (x + 1) / 2 * (max - min) + min.

    Constant channels are normalized to 0 and therefore come back as ``min``.
    The published Sharpa mask is all true, and the training inverse does not
    consult it, so this side does not either.
    """
    x = np.clip(np.asarray(value, dtype=np.float32), -1.0, 1.0)
    low = np.asarray(stats["min"], dtype=np.float32)
    high = np.asarray(stats["max"], dtype=np.float32)
    if x.shape[-1] != low.shape[0] or high.shape != low.shape:
        raise ValueError(f"action dimension {x.shape[-1]} != statistics dimension {low.shape[0]}")
    return (0.5 * (x + 1.0) * (high - low) + low).astype(np.float32, copy=False)


_EMBODIMENT_IDS = {"sharpa_left": 29, "sharpa_right": 30, "sharpa_bimanual": 16}
# Training stacks wrist views after the primary view, left then right.
_WRIST_VIEW_COUNTS = {"sharpa_left": 1, "sharpa_right": 1, "sharpa_bimanual": 2}


class SharpaLaWAMPolicy:
    def __init__(self, checkpoint: str, *, device: str, unnorm_key: str | None):
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.policy = baseframework.from_pretrained(self.checkpoint).to(device).eval()
        stats = getattr(self.policy, "norm_stats", None) or {}
        if unnorm_key is None:
            if len(stats) != 1:
                raise ValueError(f"checkpoint has multiple statistics keys {list(stats)}; pass --unnorm-key")
            unnorm_key = next(iter(stats))
        if unnorm_key not in stats:
            raise ValueError(f"unknown --unnorm-key {unnorm_key!r}; choose from {list(stats)}")
        self.unnorm_key = unnorm_key
        self.stats = stats[unnorm_key]
        try:
            self.embodiment_id = _EMBODIMENT_IDS[unnorm_key]
            self.wrist_view_count = _WRIST_VIEW_COUNTS[unnorm_key]
        except KeyError as exc:
            raise ValueError(f"No DexArena embodiment id is defined for statistics key {unnorm_key!r}") from exc
        self._lock = threading.Lock()
        flow = self.policy.policy_cfg.flow_cfg
        self.action_hz = 20.0
        self.horizon_sec = float(flow.horizon_sec)
        self.action_dim = int(flow.action_dim)
        self.state_dim = int(flow.state_dim)

    def predict_action(self, *, examples, **kwargs):
        if not isinstance(examples, list) or not examples:
            raise ValueError("examples must be a non-empty list")
        prepared = []
        for item in examples:
            ex = dict(item)
            ex["primary_image"] = [np.asarray(x, dtype=np.uint8) for x in ex["primary_image"]]
            wrist = ex.get("wrist_image")
            if not isinstance(wrist, (list, tuple)) or len(wrist) != self.wrist_view_count:
                raise ValueError(
                    f"{self.unnorm_key} was trained with {self.wrist_view_count} wrist view(s) "
                    f"after the primary view; got {0 if wrist is None else len(wrist) if isinstance(wrist, (list, tuple)) else type(wrist)}"
                )
            ex["wrist_image"] = [np.ascontiguousarray(np.asarray(x, dtype=np.uint8)) for x in wrist]
            if ex.get("wm_primary_image") is not None:
                ex["wm_primary_image"] = np.asarray(ex["wm_primary_image"], dtype=np.uint8)
            if ex.get("state") is not None:
                ex["state"] = _min_max_normalize(ex["state"], self.stats["state"])
            ex["action_hz"] = float(ex.get("action_hz", self.action_hz))
            supplied_id = int(ex["embodiment_id"])
            if supplied_id != self.embodiment_id:
                raise ValueError(
                    f"request embodiment_id={supplied_id} does not match checkpoint "
                    f"{self.unnorm_key} (id {self.embodiment_id})"
                )
            ex["embodiment_id"] = supplied_id
            prepared.append(ex)
        with self._lock, torch.inference_mode():
            output = self.policy.predict_action(
                examples=prepared,
                guidance_scale=kwargs.get("guidance_scale"),
                num_inference_steps=kwargs.get("num_inference_steps"),
            )
        normalized = np.asarray(output["normalized_actions"], dtype=np.float32)
        raw = _min_max_unnormalize(normalized, self.stats["action"])
        return {"normalized_actions": normalized, "raw_actions": raw}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to pytorch_model.pt")
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10093)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    policy = SharpaLaWAMPolicy(args.checkpoint, device=args.device, unnorm_key=args.unnorm_key)
    metadata = {
        "checkpoint": policy.checkpoint,
        "unnorm_key": policy.unnorm_key,
        "action_hz": policy.action_hz,
        "horizon_sec": policy.horizon_sec,
        "action_horizon": math.floor(policy.horizon_sec * policy.action_hz),
        "action_dim": policy.action_dim,
        "state_dim": policy.state_dim,
        "embodiment_id": policy.embodiment_id,
        "wrist_view_count": policy.wrist_view_count,
        "action_format": "raw Sharpa controller_qpos targets",
    }
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
