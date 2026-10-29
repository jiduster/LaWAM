"""DexArena Sharpa-hand contract for LaWAM.

The published datasets under ``/mnt/ceph2/DexArena/data/sharpa_hand`` store a raw
``obs_mode=state`` vector. Its leading block is joint position, but that block is
not in ``pd_joint_pos`` controller order. Actions already are. Training therefore
gathers the joint block into controller order and replaces the env-id task string
with ``utils.dex_tasks.dataset_episode_instruction``.

The gather indices were measured by matching each absolute joint target to the
next frame's state on the published recordings. The same 29-index map holds for
left and right single-arm tasks. The 58-index map is the dual-arm package task.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

# Controller channel i is read from this index of the leading qpos block.
_SINGLE_ARM_QPOS_INDEX = (
    0, 1, 2, 3, 4, 5, 6, 7,
    12, 17, 22, 27, 8, 13, 18, 23, 9, 14, 19, 24, 10, 15, 20, 25, 11, 16, 21, 26, 28,
)
_DUAL_ARM_QPOS_INDEX = (
    0, 2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11, 13,
    14, 24, 34, 44, 54, 15, 25, 35, 45, 16, 26, 36, 46, 17, 27, 37, 47, 18, 28, 38, 48, 55,
    19, 29, 39, 49, 56, 20, 30, 40, 50, 21, 31, 41, 51, 22, 32, 42, 52, 23, 33, 43, 53, 57,
)


def controller_qpos_indices(width: int) -> np.ndarray:
    """Indices that reorder a leading qpos block into controller order."""
    if width == 29:
        index = _SINGLE_ARM_QPOS_INDEX
    elif width == 58:
        index = _DUAL_ARM_QPOS_INDEX
    else:
        raise ValueError(
            f"Sharpa controller qpos width must be 29 or 58, got {width}."
        )
    array = np.asarray(index, dtype=np.int64)
    if array.shape != (width,) or set(array.tolist()) != set(range(width)):
        raise RuntimeError(f"Sharpa qpos index map for width {width} is not a permutation.")
    return array


def reorder_controller_qpos(values: np.ndarray) -> np.ndarray:
    """Gather the last axis of a qpos array into controller order."""
    array = np.asarray(values)
    if array.shape[-1] not in (29, 58):
        return array
    return np.ascontiguousarray(array[..., controller_qpos_indices(int(array.shape[-1]))])


def is_sharpa_robot(robot_type: str | None) -> bool:
    return "sharpa" in str(robot_type or "").lower()


def env_id_from_dataset_name(dataset_name: str) -> str:
    name = str(dataset_name)
    prefix = "env_"
    marker = "__obj_"
    if not name.startswith(prefix) or marker not in name:
        raise ValueError(
            f"Sharpa dataset name `{name}` is not `env_<env_id>__obj_<id>`."
        )
    return name[len(prefix):name.rindex(marker)]


def _dexarena_root() -> Path:
    configured = os.environ.get("DEXARENA_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path("/data/home/zyh/DexArena")


@lru_cache(maxsize=1)
def _dex_tasks():
    path = _dexarena_root() / "utils" / "dex_tasks.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"DexArena instruction table not found at {path}. "
            "Set DEXARENA_ROOT to the DexArena checkout."
        )
    spec = importlib.util.spec_from_file_location("lawam_dexarena_dex_tasks", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DexArena instruction table from {path}.")
    module = importlib.util.module_from_spec(spec)
    # dataclass looks the defining module up in sys.modules while the class body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def episode_instruction(dataset_name: str, episode_index: int) -> str:
    """Official recorded-episode instruction for one Sharpa dataset episode."""
    env_id = env_id_from_dataset_name(dataset_name)
    return str(_dex_tasks().dataset_episode_instruction(env_id, int(episode_index)))


def _robot_type_for_task(task) -> str:
    if bool(task.is_dual_arm):
        return "sharpa_bimanual"
    robot = str(task.robot_uids)
    if "left" in robot and "right" not in robot:
        return "sharpa_left"
    if "right" in robot and "left" not in robot:
        return "sharpa_right"
    raise ValueError(f"Cannot assign a LaWAM robot type for {task.env_id} ({robot}).")


def named_mixtures() -> dict[str, list[tuple[str, float, str]]]:
    """Dataset mixtures keyed the same way as ``DATASET_NAMED_MIXTURES``."""
    tasks = tuple(_dex_tasks().DEX_TASKS)
    entries = []
    for task in tasks:
        entries.append((task.demo_dir_name, 1.0, _robot_type_for_task(task)))
    single = [item for item in entries if item[2] != "sharpa_bimanual"]
    dual = [item for item in entries if item[2] == "sharpa_bimanual"]
    door = [
        item for item in entries
        if item[0] == "env_CabinetPourDoorOpen-v1__obj_000"
    ]
    if len(door) != 1 or not single or len(dual) != 1:
        raise RuntimeError(
            "DexArena Sharpa registry did not produce the expected door / single / dual mixtures."
        )
    return {
        "sharpa_cabinet_pour_door_open": door,
        "sharpa_single": single,
        "sharpa_bimanual": dual,
    }
