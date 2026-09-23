from __future__ import annotations

import argparse
import json
import socketserver
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
STARVLA_ROOT = SCRIPT_DIR.parents[2]
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

from examples.RoboCasa365.eval_files.robocasa365_common import POLICY_ACTION_ORDER
from starVLA.model.framework.base_framework import baseframework


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _arrayify_policy_example(example: dict[str, Any]) -> dict[str, Any]:
    converted = dict(example)
    for key in ("primary_image", "wrist_image"):
        images = converted.get(key, None)
        if images is None:
            continue
        if not isinstance(images, list):
            raise ValueError(f"`{key}` must be a list.")
        converted[key] = [np.asarray(image, dtype=np.uint8) for image in images]
    if converted.get("wm_primary_image", None) is not None:
        converted["wm_primary_image"] = np.asarray(converted["wm_primary_image"], dtype=np.uint8)
    for key in ("state", "state_mask"):
        if key in converted and converted[key] is not None:
            converted[key] = np.asarray(converted[key], dtype=np.float32)
    return converted


def _normalize_minmax(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    high_key = "max" if "max" in stats else "q99"
    low_key = "min" if "min" in stats else "q01"
    high = np.asarray(stats[high_key], dtype=np.float32)
    low = np.asarray(stats[low_key], dtype=np.float32)
    if arr.shape[-1] != int(high.shape[0]):
        raise ValueError(f"Dim mismatch: value_dim={arr.shape[-1]}, stats_dim={int(high.shape[0])}.")
    denom = high - low
    out = arr.copy()
    valid = np.abs(denom) > 1e-12
    out[..., valid] = (out[..., valid] - low[valid]) / denom[valid] * 2.0 - 1.0
    out[..., ~valid] = 0.0
    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


def _unnormalize_panda_actions(normalized_actions: np.ndarray, action_stats: dict[str, Any]) -> np.ndarray:
    normalized = np.asarray(normalized_actions, dtype=np.float32)
    if normalized.ndim == 1:
        normalized = normalized[None, :]
    high_key = "max" if "max" in action_stats else "q99"
    low_key = "min" if "min" in action_stats else "q01"
    high = np.asarray(action_stats[high_key], dtype=np.float32)
    low = np.asarray(action_stats[low_key], dtype=np.float32)
    if normalized.shape[-1] != int(high.shape[0]):
        raise ValueError(
            f"Panda action dim mismatch: action_dim={normalized.shape[-1]}, stats_dim={int(high.shape[0])}."
        )
    mask = np.asarray(action_stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    clipped = np.clip(normalized, -1.0, 1.0)
    raw = 0.5 * (clipped + 1.0) * (high - low) + low
    out = np.where(mask, raw, clipped)

    # Training uses binary normalization for gripper_close and control_mode. The
    # controller/action logs use signed commands, so map binary predictions back
    # to the observed raw signs instead of leaving them as 0/1.
    out[..., 6] = np.where(clipped[..., 6] > 0.5, 1.0, -1.0)
    out[..., 11] = np.where(clipped[..., 11] > 0.5, 1.0, -1.0)
    return out.astype(np.float32, copy=False)


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            raw = self.rfile.readline()
            if not raw:
                break
            try:
                request = json.loads(raw.decode("utf-8"))
                response = self.server.bridge.handle_request(request)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover
                response = {"ok": False, "error": {"message": str(exc)}}
            self.wfile.write(json.dumps(_to_jsonable(response), ensure_ascii=False).encode("utf-8") + b"\n")
            self.wfile.flush()


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, *args, bridge, **kwargs):
        super().__init__(*args, **kwargs)
        self.bridge = bridge


class RoboCasa365PandaBridge:
    def __init__(
        self,
        ckpt_path: str | Path,
        *,
        device: str = "cuda",
        use_bf16: bool = True,
        unnorm_key: str | None = None,
    ) -> None:
        self.ckpt_path = Path(ckpt_path).expanduser().resolve()
        self.policy = baseframework.from_pretrained(str(self.ckpt_path))
        if use_bf16:
            self.policy = self.policy.to(torch.bfloat16)
        self.policy = self.policy.to(device).eval()
        self._lock = threading.Lock()
        self.unnorm_key = self._resolve_unnorm_key(unnorm_key)
        self.dataset_stats = self.policy.norm_stats[self.unnorm_key]
        self.state_stats = self.dataset_stats["state"]
        self.action_stats = self.dataset_stats["action"]
        self.metadata = {
            "ckpt_path": str(self.ckpt_path),
            "unnorm_key": self.unnorm_key,
            "data_mix": str(getattr(self.policy.config.datasets.vla_data, "data_mix", "")),
            "action_hz": 20.0,
            "embodiment_id": 4,
            "action_format": "raw_policy_order",
            "policy_action_order": POLICY_ACTION_ORDER,
            "primary_views": ["robot0_agentview_right", "robot0_agentview_left"],
            "wrist_views": ["robot0_eye_in_hand"],
            "wm_primary_view": "robot0_agentview_right",
        }

    def _resolve_unnorm_key(self, unnorm_key: str | None) -> str:
        norm_stats = getattr(self.policy, "norm_stats", None) or {}
        if unnorm_key is None:
            if "panda_omron" in norm_stats:
                return "panda_omron"
            if len(norm_stats) != 1:
                raise ValueError(f"Please pass `--unnorm_key` from {list(norm_stats.keys())}.")
            return next(iter(norm_stats))
        if unnorm_key not in norm_stats:
            raise ValueError(f"Unknown unnorm_key `{unnorm_key}`; available keys: {list(norm_stats.keys())}.")
        return unnorm_key

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        req_type = str(request.get("type", "predict_action"))
        if req_type == "meta":
            return {"ok": True, "data": self.metadata}
        if req_type != "predict_action":
            return {"ok": False, "error": {"message": f"unsupported request type: {req_type}"}}
        examples = request.get("examples", None)
        if not isinstance(examples, list) or len(examples) == 0:
            return {"ok": False, "error": {"message": "`examples` must be a non-empty list"}}
        examples = [_arrayify_policy_example(example) for example in examples]
        for example in examples:
            example["action_hz"] = float(example.get("action_hz", 20.0))
            example["embodiment_id"] = int(example.get("embodiment_id", 4))
            if "state" in example and example["state"] is not None:
                example["state"] = _normalize_minmax(example["state"], self.state_stats)
        with self._lock:
            policy_out = self.policy.predict_action(
                examples=examples,
                return_intermediates=bool(request.get("return_intermediates", False)),
            )
        normalized_actions = np.asarray(policy_out["normalized_actions"], dtype=np.float32)
        raw_actions = _unnormalize_panda_actions(normalized_actions, self.action_stats)
        return {
            "ok": True,
            "data": _to_jsonable(
                {
                    "raw_actions": raw_actions,
                    "normalized_actions": normalized_actions,
                    "intermediates": policy_out.get("intermediates", None),
                }
            ),
        }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--unnorm_key", type=str, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6135)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_bf16", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    bridge = RoboCasa365PandaBridge(
        args.ckpt_path,
        device=args.device,
        use_bf16=bool(args.use_bf16),
        unnorm_key=args.unnorm_key,
    )
    server = _ThreadingTCPServer((args.host, int(args.port)), _RequestHandler, bridge=bridge)
    print(json.dumps({"ok": True, "data": bridge.metadata}, ensure_ascii=False), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
