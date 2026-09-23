from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
STARVLA_ROOT = SCRIPT_DIR.parents[2]
DEXWM_ROOT = Path("/data/home/zyh/dexwm/robot_sim_dexwm/robocasa-murp")
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))
if str(DEXWM_ROOT) not in sys.path:
    sys.path.insert(0, str(DEXWM_ROOT))

from examples.DexWM.eval_files.dexwm_common import build_dexwm_example


def _patch_numba_cache() -> None:
    import numba

    orig_jit = numba.jit

    def jit_no_cache(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["cache"] = False
        return orig_jit(*args, **kwargs)

    numba.jit = jit_no_cache
    os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")


def _patch_robocasa_object_sites() -> None:
    """Backfill RoboCasa object metadata sites from DexWM object bbox geoms."""

    if getattr(ET, "_dexwm_sites_patch_applied", False):
        return
    orig_parse = ET.parse

    def _has_site(root: ET.Element, name: str) -> bool:
        return any(elem.tag == "site" and elem.get("name") == name for elem in root.iter())

    def _site_parent(root: ET.Element) -> ET.Element | None:
        worldbody = root.find("worldbody")
        if worldbody is None:
            return None
        return worldbody.find("body")

    def _add_site(parent: ET.Element, name: str, pos: np.ndarray) -> None:
        ET.SubElement(
            parent,
            "site",
            {
                "name": name,
                "pos": " ".join(f"{float(v):.9g}" for v in pos),
                "rgba": "0 0 0 0",
                "size": "0.005",
            },
        )

    def parse_with_bbox_sites(source: Any, parser: Any = None) -> ET.ElementTree:
        tree = orig_parse(source, parser=parser)
        path = os.fspath(source) if isinstance(source, (str, bytes, os.PathLike)) else ""
        if not path.endswith(".xml") or "/objects/" not in path:
            return tree

        root = tree.getroot()
        required = {"bottom_site", "top_site", "horizontal_radius_site"}
        if all(_has_site(root, name) for name in required):
            return tree

        bbox = next(
            (
                elem
                for elem in root.iter("geom")
                if elem.get("name") == "reg_bbox" and elem.get("type") == "box"
            ),
            None,
        )
        if bbox is None:
            return tree
        parent = _site_parent(root)
        if parent is None:
            return tree

        center = np.fromstring(bbox.get("pos", "0 0 0"), sep=" ", dtype=np.float64)
        half_size = np.fromstring(bbox.get("size", "0 0 0"), sep=" ", dtype=np.float64)
        if center.shape[0] != 3 or half_size.shape[0] != 3 or not np.all(np.isfinite(half_size)):
            return tree

        if not _has_site(root, "bottom_site"):
            _add_site(parent, "bottom_site", center + np.array([0.0, 0.0, -half_size[2]]))
        if not _has_site(root, "top_site"):
            _add_site(parent, "top_site", center + np.array([0.0, 0.0, half_size[2]]))
        if not _has_site(root, "horizontal_radius_site"):
            _add_site(parent, "horizontal_radius_site", center + np.array([half_size[0], half_size[1], 0.0]))
        return tree

    ET.parse = parse_with_bbox_sites
    setattr(ET, "_dexwm_sites_patch_applied", True)


def _load_json_attr(attrs: Any, key: str) -> dict[str, Any] | None:
    if key not in attrs:
        return None
    raw = attrs[key]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _resolve_task_description(ep_meta: Any, fallback: str = "dexwm task") -> str:
    if ep_meta is None:
        return str(fallback)
    if isinstance(ep_meta, bytes):
        ep_meta = ep_meta.decode("utf-8")
    try:
        parsed = json.loads(ep_meta)
    except Exception:
        return str(fallback)
    if not isinstance(parsed, dict):
        return str(fallback)
    return str(parsed.get("free_form_lang") or parsed.get("lang") or fallback)


def _load_gt_actions(demo: Any, action_source: str) -> np.ndarray | None:
    key_by_source = {
        "gt_abs_right_arm_base": "abs_right_arm_base_action",
        "gt_abs_action": "abs_action",
        "gt_actions": "actions",
    }
    key = key_by_source.get(str(action_source), None)
    if key is None:
        return None
    if key not in demo:
        raise KeyError(f"Requested action source `{key}` is missing from episode.")
    actions = np.asarray(demo[key], dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected `{key}` to have shape [T, D], got {actions.shape}.")
    return actions


def _configure_controller_for_action_source(env_kwargs: dict[str, Any], action_source: str) -> None:
    controller_configs = env_kwargs.get("controller_configs", None)
    if not isinstance(controller_configs, dict):
        return

    right_cfg = controller_configs.get("body_parts", {}).get("right", None)
    if not isinstance(right_cfg, dict):
        return

    if action_source in {"policy", "gt_abs_right_arm_base"}:
        controller_configs["control_delta"] = False
        right_cfg["input_type"] = "absolute"
        right_cfg["input_ref_frame"] = "right_base"
    elif action_source == "gt_abs_action":
        controller_configs["control_delta"] = False
        right_cfg["input_type"] = "absolute"
        right_cfg["input_ref_frame"] = "base"


def _obs_image(obs: dict[str, Any], key: str) -> np.ndarray | None:
    nested = obs.get("observation", obs)
    image = nested.get(key, None)
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim != 3:
        return None
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    # RoboCasa offscreen images are vertically flipped relative to normal viewing.
    return image[::-1].copy()


def _env_diagnostics(env: Any, step: int, action: np.ndarray, success: bool) -> dict[str, Any]:
    diag: dict[str, Any] = {
        "step": int(step),
        "success": bool(success),
        "action_l2": float(np.linalg.norm(action)),
        "action_min": float(np.min(action)),
        "action_max": float(np.max(action)),
    }
    try:
        if "obj" in getattr(env, "obj_body_id", {}):
            body_id = env.obj_body_id["obj"]
            diag["obj_pos"] = np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).tolist()
    except Exception:
        pass
    try:
        diag["right_eef_pos"] = np.asarray(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]).tolist()
    except Exception:
        pass
    return diag


def _resolve_object_position(env: Any) -> np.ndarray | None:
    try:
        if "obj" in getattr(env, "obj_body_id", {}):
            body_id = env.obj_body_id["obj"]
            return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).copy()
    except Exception:
        pass
    return None


def _write_rollout_outputs(
    output_dir: Path,
    *,
    robotview_frames: list[np.ndarray],
    wrist_frames: list[np.ndarray],
    actions: list[np.ndarray],
    diagnostics: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, str]:
    import imageio.v2 as imageio

    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    if robotview_frames:
        path = output_dir / "robotview.mp4"
        imageio.mimsave(path, robotview_frames, fps=5)
        written["robotview_video"] = str(path)
    if wrist_frames:
        path = output_dir / "wrist.mp4"
        imageio.mimsave(path, wrist_frames, fps=5)
        written["wrist_video"] = str(path)

    actions_path = output_dir / "actions.npy"
    np.save(actions_path, np.asarray(actions, dtype=np.float32))
    written["actions"] = str(actions_path)

    diagnostics_path = output_dir / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    written["diagnostics"] = str(diagnostics_path)

    summary_path = output_dir / "summary.json"
    summary_with_outputs = dict(summary)
    summary_with_outputs["outputs"] = written
    summary_path.write_text(json.dumps(summary_with_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    written["summary_json"] = str(summary_path)
    return written


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=str, required=True)
    parser.add_argument("--policy_host", type=str, default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=5694)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--task_description", type=str, default="")
    parser.add_argument(
        "--action_source",
        type=str,
        default="policy",
        choices=("policy", "gt_abs_right_arm_base", "gt_abs_action", "gt_actions"),
        help="Use model policy actions, or replay a ground-truth action array from the HDF5 episode.",
    )
    parser.add_argument(
        "--stop_on_object_moved",
        type=float,
        default=0.0,
        help="Stop early when object displacement from reset exceeds this threshold in meters. Disabled when <=0.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser


def main() -> None:
    _patch_numba_cache()
    import robocasa  # noqa: F401
    import robosuite
    import robocasa.scripts.playback_utils as P

    from examples.DexWM.eval_files.model2dexwm_interface import ModelClient

    args = build_argparser().parse_args()
    _patch_robocasa_object_sites()
    model = None
    if args.action_source == "policy":
        model = ModelClient(
            policy_ckpt_path=os.environ["POLICY_CKPT_PATH"],
            host=args.policy_host,
            port=args.policy_port,
            bridge_mode="socket",
        )

    with h5py.File(args.dataset, "r") as f:
        demo = f["data"][args.episode]
        gt_actions = _load_gt_actions(demo, args.action_source)
        env_meta = _load_json_attr(f["data"].attrs, "env_args") or {}
        env_kwargs = dict(env_meta.get("env_kwargs", {}))
        env_kwargs["env_name"] = env_meta.get("env_name")
        env_kwargs["has_renderer"] = False
        env_kwargs["has_offscreen_renderer"] = True
        env_kwargs["use_camera_obs"] = True
        env_kwargs["camera_depths"] = False
        env_kwargs.setdefault("control_freq", 1)
        _configure_controller_for_action_source(env_kwargs, args.action_source)
        env = robosuite.make(**env_kwargs)

        initial_state = {
            "states": np.asarray(demo["states"][0]),
            "model": demo.attrs["model_file"],
            "ep_meta": demo.attrs.get("ep_meta", None),
        }
        P.reset_to(env, initial_state)

        obs = env._get_observations(force_update=True)
        initial_obj_pos = _resolve_object_position(env)
        task_description = args.task_description.strip() or _resolve_task_description(demo.attrs.get("ep_meta", None))
        rollout_actions: list[np.ndarray] = []
        diagnostics: list[dict[str, Any]] = []
        robotview_frames: list[np.ndarray] = []
        wrist_frames: list[np.ndarray] = []
        for step in range(int(args.steps)):
            robotview = _obs_image(obs, "robot0_robotview_2_image")
            wrist = _obs_image(obs, "gripper0_right_right_eye_in_hand_image")
            if robotview is not None:
                robotview_frames.append(robotview)
            if wrist is not None:
                wrist_frames.append(wrist)

            if args.action_source == "policy":
                if model is None:
                    raise RuntimeError("Policy model client was not initialized.")
                example = build_dexwm_example(
                    task_description,
                    obs,
                    wm_primary_video_key=getattr(model, "wm_primary_video_key", None),
                )
                action = model.step(example, step=step, slot_id=0)
            else:
                if gt_actions is None:
                    raise RuntimeError("Ground-truth actions were not loaded.")
                action = gt_actions[min(step, int(gt_actions.shape[0]) - 1)]
                if action.shape[0] != int(env.action_dim):
                    raise ValueError(
                        f"Ground-truth action dim mismatch: action_dim={action.shape[0]}, env_dim={env.action_dim}."
                    )
            rollout_actions.append(np.asarray(action, dtype=np.float32))
            obs, _, _, _ = env.step(action)
            success_now = bool(env._check_success())
            diag = _env_diagnostics(env, step, np.asarray(action, dtype=np.float32), success_now)
            stop_reason = None
            if initial_obj_pos is not None and float(args.stop_on_object_moved) > 0:
                obj_pos = _resolve_object_position(env)
                if obj_pos is not None:
                    obj_displacement = float(np.linalg.norm(obj_pos - initial_obj_pos))
                    diag["obj_displacement_from_reset"] = obj_displacement
                    if obj_displacement > float(args.stop_on_object_moved):
                        stop_reason = "object_moved_too_far"
                        diag["stop_reason"] = stop_reason
            diagnostics.append(diag)
            if success_now:
                break
            if stop_reason is not None:
                break

        robotview = _obs_image(obs, "robot0_robotview_2_image")
        wrist = _obs_image(obs, "gripper0_right_right_eye_in_hand_image")
        if robotview is not None:
            robotview_frames.append(robotview)
        if wrist is not None:
            wrist_frames.append(wrist)

        summary = {
            "success": bool(env._check_success()),
            "steps": len(rollout_actions),
            "action_source": str(args.action_source),
            "right_input_ref_frame": str(
                env_kwargs.get("controller_configs", {})
                .get("body_parts", {})
                .get("right", {})
                .get("input_ref_frame", "")
            ),
        }
        if diagnostics and "stop_reason" in diagnostics[-1]:
            summary["stop_reason"] = diagnostics[-1]["stop_reason"]
        if args.output_dir is not None:
            summary["outputs"] = _write_rollout_outputs(
                args.output_dir,
                robotview_frames=robotview_frames,
                wrist_frames=wrist_frames,
                actions=rollout_actions,
                diagnostics=diagnostics,
                summary=summary,
            )
        print(
            json.dumps(
                summary,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
