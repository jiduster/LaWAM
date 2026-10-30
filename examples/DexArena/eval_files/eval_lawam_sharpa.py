"""Run a LaWAM Sharpa checkpoint against one DexArena task over WebSocket."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def _dexarena_imports(dexarena_root: Path, task_module: str):
    root = str(dexarena_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.utils.wrappers.record import RecordEpisode
    from utils.dex_cameras import install_camera_profile
    from utils.dex_tasks import episode_instruction, resolve_dex_task
    from utils.image_observation import (
        BASE_CAMERA_ALIASES,
        LEFT_WRIST_CAMERA_ALIASES,
        RIGHT_WRIST_CAMERA_ALIASES,
        extract_camera_rgb,
        extract_controller_qpos,
    )

    module_name = "task." + Path(task_module).with_suffix("").as_posix().replace("/", ".").removeprefix("task.")
    importlib.import_module(module_name)
    return (gym, FlattenActionSpaceWrapper, RecordEpisode,
            install_camera_profile, episode_instruction, resolve_dex_task,
            BASE_CAMERA_ALIASES, LEFT_WRIST_CAMERA_ALIASES, RIGHT_WRIST_CAMERA_ALIASES,
            extract_camera_rgb, extract_controller_qpos)


def _any_flag(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "detach"):
        return bool(value.detach().bool().reshape(-1).any().item())
    return bool(np.asarray(value).astype(bool).reshape(-1).any())


def _mean_float(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().float().mean().item())
    return float(np.asarray(value, dtype=np.float64).mean())


def _env_action(action: np.ndarray, env) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    space = env.action_space
    if action.size != int(np.prod(space.shape)):
        raise ValueError(f"model action dim {action.size} does not match env action shape {space.shape}")
    action = action.reshape(space.shape)
    return np.clip(action, space.low, space.high).astype(np.float32, copy=False)


def _wrist_views(obs, task, left_aliases, right_aliases, extract_camera_rgb) -> list[np.ndarray]:
    """Match training: one wrist for a single arm, left then right for both arms."""
    if task.is_dual_arm:
        return [
            extract_camera_rgb(obs, left_aliases),
            extract_camera_rgb(obs, right_aliases),
        ]
    if "left" in task.robot_uids:
        return [extract_camera_rgb(obs, left_aliases)]
    return [extract_camera_rgb(obs, right_aliases)]


def _make_env(args, task, api):
    gym, FlattenActionSpaceWrapper, RecordEpisode, install_camera_profile = api[:4]
    install_camera_profile(args.env_id, args.camera_profile)
    env_kwargs: dict[str, Any] = {
        "obs_mode": "rgb",
        "render_mode": "rgb_array",
        "control_mode": args.control_mode or task.control_mode,
        "reward_mode": args.reward_mode,
        "max_episode_steps": args.max_episode_steps or task.max_episode_steps,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
    }
    if args.env_extra:
        env_kwargs.update(json.loads(args.env_extra))
    # Stay on the ManiSkill env itself. A gymnasium vector wrapper replaces
    # ``unwrapped`` with itself, so controller-qpos can no longer be read.
    env = gym.make(args.env_id, num_envs=1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    if args.video_dir:
        env = RecordEpisode(
            env,
            output_dir=args.video_dir,
            save_trajectory=False,
            max_steps_per_video=args.max_episode_steps or task.max_episode_steps,
            video_fps=int(getattr(env.unwrapped, "control_freq", 20)),
        )
    return env


def evaluate(args) -> dict[str, Any]:
    dex_root = Path(args.dexarena_root).expanduser().resolve()
    api = _dexarena_imports(dex_root, args.task_module)
    (_, _, _, _, episode_instruction, resolve_dex_task, base_aliases,
     left_aliases, right_aliases, extract_camera_rgb, extract_controller_qpos) = api
    task = resolve_dex_task(args.env_id)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    try:
        metadata = client.get_server_metadata()
        expected_key = args.unnorm_key or ("sharpa_bimanual" if task.is_dual_arm else
                                           "sharpa_left" if "left" in task.robot_uids else "sharpa_right")
        expected_embodiment_id = {
            "sharpa_left": 29,
            "sharpa_right": 30,
            "sharpa_bimanual": 16,
        }[expected_key]
        if metadata.get("unnorm_key") != expected_key:
            raise ValueError(f"server statistics key is {metadata.get('unnorm_key')!r}, expected {expected_key!r}")
        if int(metadata.get("embodiment_id", -1)) != expected_embodiment_id:
            raise ValueError(
                f"server embodiment_id={metadata.get('embodiment_id')} but "
                f"{expected_key} requires {expected_embodiment_id}"
            )
        if int(metadata.get("action_dim", -1)) != task.action_dim:
            raise ValueError(f"server action_dim={metadata.get('action_dim')} but task requires {task.action_dim}")
        expected_wrists = 2 if task.is_dual_arm else 1
        if int(metadata.get("wrist_view_count", expected_wrists)) != expected_wrists:
            raise ValueError(
                f"server wrist_view_count={metadata.get('wrist_view_count')} but {expected_key} needs {expected_wrists}"
            )
        horizon = int(metadata.get("action_horizon", 0))
        if horizon < 1:
            raise ValueError(f"invalid server action_horizon in metadata: {metadata}")
        execute_steps = args.execute_steps or horizon
        if execute_steps > horizon:
            raise ValueError(f"--execute-steps {execute_steps} exceeds server chunk horizon {horizon}")
        step_limit = args.steps if args.steps is not None else (args.max_episode_steps or task.max_episode_steps)

        env = _make_env(args, task, api)
        try:
            action_space = env.action_space
            if int(np.prod(action_space.shape)) != task.action_dim:
                raise ValueError(f"DexArena action dim {action_space.shape} != task controller dim {task.action_dim}")
            episodes = []
            for episode_index in range(args.episodes):
                seed = args.seed + episode_index
                instruction = episode_instruction(args.env_id, seed)
                obs, _ = env.reset(seed=seed)
                success = False
                total_return = 0.0
                steps = 0
                done = False
                while steps < step_limit and not done:
                    primary = extract_camera_rgb(obs, base_aliases)
                    wrists = _wrist_views(obs, task, left_aliases, right_aliases, extract_camera_rgb)
                    qpos = extract_controller_qpos(obs, env=env)
                    if qpos.shape != (task.action_dim,):
                        raise ValueError(f"controller qpos shape {qpos.shape} != {(task.action_dim,)}")
                    example = {
                        "lang": instruction,
                        "primary_image": [primary],
                        "wrist_image": wrists,
                        "state": qpos,
                        "action_hz": float(metadata.get("action_hz", 20.0)),
                        "embodiment_id": int(metadata["embodiment_id"]),
                    }
                    response = client.predict_action({"examples": [example]})
                    if not response.get("ok", False):
                        raise RuntimeError(f"LaWAM inference failed: {response}")
                    data = response.get("data", {})
                    actions = np.asarray(data.get("raw_actions"), dtype=np.float32)
                    if actions.ndim == 3:
                        actions = actions[0]
                    if actions.ndim != 2 or actions.shape[1] != task.action_dim or actions.shape[0] < execute_steps:
                        raise ValueError(f"unexpected raw action chunk shape {actions.shape}; expected [T,{task.action_dim}]")
                    for action in actions[:execute_steps]:
                        obs, reward, terminated, truncated, info = env.step(_env_action(action, env))
                        total_return += _mean_float(reward)
                        success = success or _any_flag(info.get("success"))
                        steps += 1
                        done = _any_flag(terminated) or _any_flag(truncated)
                        if done or steps >= step_limit:
                            break
                record = {
                    "episode_index": episode_index,
                    "seed": seed,
                    "instruction": instruction,
                    "success": success,
                    "return": total_return,
                    "episode_length": steps,
                }
                episodes.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
            summary = {
                "env_id": args.env_id,
                "checkpoint": metadata.get("checkpoint"),
                "unnorm_key": expected_key,
                "camera_profile": args.camera_profile,
                "action_horizon": horizon,
                "execute_steps": execute_steps,
                "step_limit": step_limit,
                "num_episodes": len(episodes),
                "success_rate": float(np.mean([item["success"] for item in episodes])),
                "mean_return": float(np.mean([item["return"] for item in episodes])),
                "mean_episode_length": float(np.mean([item["episode_length"] for item in episodes])),
                "episodes": episodes,
            }
            if args.output:
                output = Path(args.output).expanduser()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, ensure_ascii=False), flush=True)
            return summary
        finally:
            env.close()
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--task-module", required=True, help="DexArena module path, e.g. task/cabinet_pour_door_open.py")
    parser.add_argument("--dexarena-root", default="/data/home/zyh/DexArena")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--execute-steps", type=int, default=None, help="steps executed from each predicted chunk; defaults to full chunk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-backend", default="physx_cuda")
    parser.add_argument("--render-backend", default="sapien_cuda")
    parser.add_argument("--control-mode", default=None)
    parser.add_argument("--reward-mode", default="sparse")
    parser.add_argument("--reconfiguration-freq", type=int, default=1)
    parser.add_argument("--camera-profile", default="robodex_fixed")
    parser.add_argument("--env-extra", default=None, help="JSON object passed as extra gym.make kwargs")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--output", default=None, help="write summary JSON, including episode records")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.max_episode_steps is not None and args.max_episode_steps < 1:
        parser.error("--max-episode-steps must be positive")
    if args.execute_steps is not None and args.execute_steps < 1:
        parser.error("--execute-steps must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
