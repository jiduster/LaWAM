from __future__ import annotations

import json
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
STARVLA_ROOT = SCRIPT_DIR.parents[2]
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

from .dexwm_common import remap_dexwm_action_to_env, resolve_dexwm_control_from_data_mix


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


def _send_request(host: str, port: int, request: dict[str, Any], timeout_sec: float = 300.0) -> dict[str, Any]:
    with socket.create_connection((host, int(port)), timeout=float(timeout_sec)) as sock:
        sock.sendall(json.dumps(_to_jsonable(request), ensure_ascii=False).encode("utf-8") + b"\n")
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    raw = b"".join(chunks).split(b"\n", 1)[0]
    return json.loads(raw.decode("utf-8"))


@dataclass
class _SlotState:
    task_description: str | None = None
    raw_actions: Optional[np.ndarray] = None
    action_cursor: int = 0

    def reset(self, task_description: str | None = None) -> None:
        self.task_description = task_description
        self.raw_actions = None
        self.action_cursor = 0

    def needs_query(self) -> bool:
        return self.raw_actions is None or self.action_cursor >= int(self.raw_actions.shape[0])


class _SocketPolicyClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = str(host)
        self.port = int(port)

    def get_server_metadata(self) -> dict[str, Any]:
        response = _send_request(self.host, self.port, {"type": "meta"})
        if not response.get("ok", False):
            raise RuntimeError(f"Failed to query server metadata: {response}")
        return dict(response.get("data", {}))

    def predict_action(self, query_info: dict[str, Any]) -> dict[str, Any]:
        response = _send_request(self.host, self.port, {"type": "predict_action", **query_info})
        if not response.get("ok", False):
            raise RuntimeError(f"Failed to query policy bridge: {response}")
        return dict(response)

    def close(self) -> None:
        return None


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        image_size: Optional[Sequence[int]] = None,
        replan_steps: Optional[int | str] = None,
        action_ensemble: Optional[bool | str] = None,
        action_ensemble_alpha: Optional[float | str] = None,
        action_reorder: Optional[Sequence[int] | str] = None,
        host: str = "127.0.0.1",
        port: int = 5694,
        bridge_mode: str = "socket",
    ) -> None:
        del action_ensemble, action_ensemble_alpha, action_reorder, replan_steps, image_size
        self.policy_ckpt_path = str(Path(policy_ckpt_path).expanduser().resolve())
        self.host = str(host)
        self.port = int(port)
        self.bridge_mode = str(bridge_mode)
        self._client = _SocketPolicyClient(self.host, self.port)
        metadata = self._client.get_server_metadata()
        if str(metadata.get("ckpt_path", "")) != self.policy_ckpt_path:
            raise ValueError("Checkpoint mismatch between client and server.")
        self.control_spec = resolve_dexwm_control_from_data_mix(metadata.get("data_mix", "dexwm_robocasa"))
        self.action_hz = float(metadata.get("action_hz", self.control_spec.action_hz))
        self.embodiment_id = int(metadata.get("embodiment_id", 31))
        self.wm_primary_video_key = str(metadata.get("wm_primary_video_key", "robot0_robotview_2_image"))
        self._slot_states: dict[int, _SlotState] = {}

    def close(self) -> None:
        self._client.close()

    def reset(self, task_description: str, slot_id: int = 0, **kwargs) -> None:
        del kwargs
        self._get_slot_state(slot_id).reset(task_description=task_description)

    def needs_query(self, slot_id: int = 0, task_description: Optional[str] = None) -> bool:
        slot_state = self._get_slot_state(slot_id)
        if task_description is not None and task_description != slot_state.task_description:
            slot_state.reset(task_description=task_description)
        return slot_state.needs_query()

    def step(self, example: dict[str, Any], step: int = 0, slot_id: int = 0) -> np.ndarray:
        del step
        return self.step_batch([example], slot_ids=[slot_id])[0]

    def step_batch(
        self,
        examples: Sequence[dict[str, Any]],
        *,
        slot_ids: Optional[Sequence[int]] = None,
    ) -> list[np.ndarray]:
        if slot_ids is None:
            slot_ids = list(range(len(examples)))
        if len(examples) != len(slot_ids):
            raise ValueError("`examples` and `slot_ids` must have the same length.")

        query_positions: list[int] = []
        query_examples: list[dict[str, Any]] = []
        outputs: list[np.ndarray | None] = [None] * len(examples)

        for idx, (example, slot_id) in enumerate(zip(examples, slot_ids)):
            slot_id = int(slot_id)
            slot_state = self._get_slot_state(slot_id)
            prepared = self._prepare_example(example)
            task_description = str(prepared["lang"])
            if task_description != slot_state.task_description:
                slot_state.reset(task_description=task_description)
            if slot_state.needs_query():
                query_positions.append(idx)
                query_examples.append(self._build_infer_example(prepared))

        if query_examples:
            response = self._client.predict_action({"examples": query_examples})
            action_payload = response["data"].get("raw_actions", response["data"].get("normalized_actions"))
            normalized_actions = np.asarray(action_payload, dtype=np.float32)
            if normalized_actions.ndim == 2:
                normalized_actions = normalized_actions[None, :, :]
            for batch_idx, example_idx in enumerate(query_positions):
                slot_id = int(slot_ids[example_idx])
                slot_state = self._get_slot_state(slot_id)
                slot_state.raw_actions = np.asarray(normalized_actions[batch_idx], dtype=np.float32)
                slot_state.action_cursor = 0

        for idx, slot_id in enumerate(slot_ids):
            slot_state = self._get_slot_state(int(slot_id))
            if slot_state.raw_actions is None or slot_state.action_cursor >= int(slot_state.raw_actions.shape[0]):
                raise RuntimeError(f"Slot {slot_id} has no cached actions.")
            current_action = np.asarray(slot_state.raw_actions[slot_state.action_cursor], dtype=np.float32)
            slot_state.action_cursor += 1
            outputs[idx] = remap_dexwm_action_to_env(current_action, self.control_spec)

        return [np.asarray(output, dtype=np.float32) for output in outputs if output is not None]

    def _get_slot_state(self, slot_id: int) -> _SlotState:
        slot_id = int(slot_id)
        if slot_id not in self._slot_states:
            self._slot_states[slot_id] = _SlotState()
        return self._slot_states[slot_id]

    def _prepare_example(self, example: dict[str, Any]) -> dict[str, Any]:
        primary_images = example.get("primary_image", None)
        wm_primary_image = example.get("wm_primary_image", None)
        wrist_images = example.get("wrist_image", None)
        legacy_images = example.get("image", None)

        if primary_images is None and legacy_images is not None:
            if not isinstance(legacy_images, (list, tuple)) or len(legacy_images) == 0:
                raise ValueError("DexWM example `image` must be a non-empty list.")
            primary_images = [legacy_images[0]]
            wrist_images = list(legacy_images[1:])

        if not isinstance(primary_images, (list, tuple)) or len(primary_images) == 0:
            raise ValueError("DexWM example must contain non-empty `primary_image` list.")
        if wrist_images is None:
            wrist_images = []
        if not isinstance(wrist_images, (list, tuple)):
            raise ValueError("DexWM example `wrist_image` must be a list when provided.")

        return {
            "lang": str(example.get("lang", "")),
            "primary_image": [np.asarray(image) for image in primary_images],
            "wm_primary_image": None if wm_primary_image is None else np.asarray(wm_primary_image),
            "wrist_image": [np.asarray(image) for image in wrist_images],
            "state": self._prepare_state(example.get("state", None)),
        }

    def _build_infer_example(self, example: dict[str, Any]) -> dict[str, Any]:
        infer_example: dict[str, Any] = {
            "lang": str(example["lang"]),
            "primary_image": list(example["primary_image"]),
            "embodiment_id": int(self.embodiment_id),
            "action_hz": float(self.action_hz),
        }
        if example.get("wm_primary_image", None) is not None:
            infer_example["wm_primary_image"] = np.asarray(example["wm_primary_image"], dtype=np.uint8)
        if len(example["wrist_image"]) > 0:
            infer_example["wrist_image"] = list(example["wrist_image"])
        if example.get("state", None) is not None:
            infer_example["state"] = np.asarray(example["state"], dtype=np.float32)
        return infer_example

    def _prepare_state(self, state: Any) -> Optional[np.ndarray]:
        if state is None:
            return None
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.ndim == 2 and state_array.shape[0] == 1:
            state_array = state_array[0]
        if state_array.ndim != 1:
            raise ValueError(f"`state` must have shape [D], got {state_array.shape}.")
        return state_array
