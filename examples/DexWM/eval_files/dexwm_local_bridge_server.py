from __future__ import annotations

import argparse
import json
import socketserver
import threading
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
STARVLA_ROOT = SCRIPT_DIR.parents[2]
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

from examples.DexWM.eval_files.dexwm_common import unnormalize_dexwm_actions
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


def _normalize_dexwm_state(state: np.ndarray, state_stats: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32)
    high_key = "max" if "max" in state_stats else "q99"
    low_key = "min" if "min" in state_stats else "q01"
    high = np.asarray(state_stats[high_key], dtype=np.float32)
    low = np.asarray(state_stats[low_key], dtype=np.float32)
    if arr.shape[-1] != int(high.shape[0]):
        raise ValueError(
            f"DexWM state dim mismatch: state_dim={arr.shape[-1]}, stats_dim={int(high.shape[0])}."
        )
    denom = high - low
    out = arr.copy()
    valid = np.abs(denom) > 1e-12
    out[..., valid] = (out[..., valid] - low[valid]) / denom[valid] * 2.0 - 1.0
    out[..., ~valid] = 0.0
    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            raw = self.rfile.readline()
            if not raw:
                break
            try:
                request = json.loads(raw.decode("utf-8"))
                response = self.server.bridge.handle_request(request)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive
                response = {"ok": False, "error": {"message": str(exc)}}
            self.wfile.write(json.dumps(_to_jsonable(response), ensure_ascii=False).encode("utf-8") + b"\n")
            self.wfile.flush()


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, *args, bridge, **kwargs):
        super().__init__(*args, **kwargs)
        self.bridge = bridge


class DexWMLocalBridge:
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
        self.state_stats = self.policy.norm_stats[self.unnorm_key]["state"]
        self.metadata = {
            "ckpt_path": str(self.ckpt_path),
            "framework_name": str(getattr(self.policy.config.framework, "name", "")),
            "data_mix": str(getattr(self.policy.config.datasets.vla_data, "data_mix", "dexwm_robocasa")),
            "action_hz": float(getattr(self.policy.config.datasets.vla_data, "action_hz_override", 1.0) or 1.0),
            "wm_primary_video_key": str(
                getattr(self.policy.config.datasets.vla_data, "wm_primary_video_key", "robot0_robotview_2_image")
            ),
            "embodiment_id": 31,
            "unnorm_key": self.unnorm_key,
            "action_format": "raw_unnormalized",
        }

    def _resolve_unnorm_key(self, unnorm_key: str | None) -> str:
        norm_stats = getattr(self.policy, "norm_stats", None) or {}
        if unnorm_key is None:
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
            if "state" in example and example["state"] is not None:
                example["state"] = _normalize_dexwm_state(example["state"], self.state_stats)
        with self._lock:
            policy_out = self.policy.predict_action(
                examples=examples,
                return_intermediates=bool(request.get("return_intermediates", False)),
            )
        normalized_actions = policy_out["normalized_actions"]
        action_stats = self.policy.get_action_stats(unnorm_key=self.unnorm_key, norm_stats=self.policy.norm_stats)
        unnormalized_actions = unnormalize_dexwm_actions(np.asarray(normalized_actions, dtype=np.float32), action_stats)
        return {
            "ok": True,
            "data": _to_jsonable(
                {
                    "raw_actions": unnormalized_actions,
                    "normalized_actions": unnormalized_actions,
                    "intermediates": policy_out.get("intermediates", None),
                }
            ),
        }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--unnorm_key", type=str, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_bf16", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    bridge = DexWMLocalBridge(
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
