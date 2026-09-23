from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import re
import tempfile
import socket
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
STARVLA_ROOT = SCRIPT_DIR.parents[2]
DEXWM_ROOT = Path("/data/home/zyh/dexwm/robot_sim_dexwm/robocasa-murp")
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))
if str(DEXWM_ROOT) not in sys.path:
    sys.path.insert(0, str(DEXWM_ROOT))

from examples.RoboCasa365.eval_files.robocasa365_common import POLICY_ACTION_ORDER


POLICY_TO_SIM_CAMERA = {
    "robot0_agentview_right": "robot0_robotview",
    "robot0_eye_in_hand": "robot0_eye_in_hand",
    "robot0_agentview_left": "robot0_frontview",
}
VIDEO_CAMERA_NAMES = ["robot0_robotview", "robot0_eye_in_hand", "robot0_frontview"]


def _patch_numba_cache() -> None:
    import numba

    if getattr(numba, "_lawam_no_cache_patch", False):
        return
    orig_jit = numba.jit

    def jit_no_cache(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["cache"] = False
        return orig_jit(*args, **kwargs)

    numba.jit = jit_no_cache
    setattr(numba, "_lawam_no_cache_patch", True)
    os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")


def _patch_robocasa_object_sites() -> None:
    if getattr(ET, "_lawam_sites_patch_applied", False):
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
        if center.shape[0] != 3 or half_size.shape[0] != 3:
            return tree
        if not _has_site(root, "bottom_site"):
            _add_site(parent, "bottom_site", center + np.array([0.0, 0.0, -half_size[2]]))
        if not _has_site(root, "top_site"):
            _add_site(parent, "top_site", center + np.array([0.0, 0.0, half_size[2]]))
        if not _has_site(root, "horizontal_radius_site"):
            _add_site(parent, "horizontal_radius_site", center + np.array([half_size[0], half_size[1], 0.0]))
        return tree

    ET.parse = parse_with_bbox_sites
    setattr(ET, "_lawam_sites_patch_applied", True)


def _patch_right_base_site_fallback() -> None:
    import robocasa.scripts.playback_utils as P

    if getattr(P, "_lawam_right_base_patch", False):
        return
    orig = P.get_ee_T_arm_base

    def get_ee_T_arm_base_fallback(env, ee_T_base_pos_quat, arm="right", mode="playback"):
        try:
            return orig(env, ee_T_base_pos_quat, arm=arm, mode=mode)
        except Exception as exc:
            if arm != "right" or "right_base_center" not in str(exc):
                raise
            import robosuite.utils.transform_utils as T

            data = env.sim.data if mode == "playback" else env.data
            arm_base_T_world_pos = data.get_site_xpos("robot0_right_center")
            arm_base_T_world_mat = data.get_site_xmat("robot0_right_center")
            base_T_world_pos = data.get_body_xpos("mobilebase0_base")
            base_T_world_mat = data.get_body_xmat("mobilebase0_base")
            base_T_world = np.eye(4)
            base_T_world[:3, -1] = base_T_world_pos
            base_T_world[:3, :3] = base_T_world_mat
            ee_T_base_pos = np.asarray(ee_T_base_pos_quat[0:3], dtype=np.float64)
            ee_T_base_mat = T.quat2mat(T.axisangle2quat(ee_T_base_pos_quat[3:]))
            ee_T_base = np.eye(4)
            ee_T_base[:3, -1] = ee_T_base_pos
            ee_T_base[:3, :3] = ee_T_base_mat
            ee_T_world = base_T_world @ ee_T_base
            arm_base_T_world = np.eye(4)
            arm_base_T_world[:3, -1] = arm_base_T_world_pos
            arm_base_T_world[:3, :3] = arm_base_T_world_mat
            ee_T_arm = np.linalg.inv(arm_base_T_world) @ ee_T_world
            ee_T_arm_pos = ee_T_arm[:3, -1]
            ee_T_arm_quat = T.mat2quat(ee_T_arm[:3, :3])
            ee_T_arm_axisangle = T.quat2axisangle(ee_T_arm_quat)
            return np.concatenate((ee_T_arm_pos, ee_T_arm_axisangle)).astype(np.float32)

    P.get_ee_T_arm_base = get_ee_T_arm_base_fallback
    setattr(P, "_lawam_right_base_patch", True)


def _patch_mjcf_object_tmp_writes() -> None:
    from robocasa.models.objects.objects import MJCFObject
    from robosuite.models.objects import MujocoXMLObject

    if getattr(MJCFObject, "_lawam_tmp_write_patch", False):
        return

    def _absolutize_asset_paths(root: ET.Element, xml_path: str) -> None:
        folder = os.path.dirname(os.path.abspath(xml_path))
        asset = root.find("asset")
        if asset is None:
            return
        for elem in list(asset.findall("mesh")) + list(asset.findall("texture")):
            old_path = elem.get("file")
            if old_path and not os.path.isabs(old_path):
                elem.set("file", os.path.normpath(os.path.join(folder, old_path)))

    def patched_init(
        self,
        name,
        mjcf_path,
        scale=1.0,
        solimp=(0.998, 0.998, 0.001),
        solref=(0.001, 1),
        density=100,
        friction=(0.95, 0.3, 0.1),
        margin=None,
        rgba=None,
        priority=None,
    ):
        if isinstance(scale, float):
            scale_arr = [scale, scale, scale]
        elif isinstance(scale, (tuple, list)):
            if len(scale) != 3:
                raise ValueError(f"got invalid scale: {scale}")
            scale_arr = tuple(scale)
        else:
            raise ValueError(f"got invalid scale: {scale}")
        scale_arr = np.array(scale_arr)

        self.solimp = solimp
        self.solref = solref
        self.density = density
        self.friction = friction
        self.margin = margin
        self.priority = priority
        self.rgba = rgba

        xml_path = os.path.abspath(os.path.expanduser(mjcf_path))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        _absolutize_asset_paths(root, xml_path)
        xml_str = ET.tostring(root, encoding="utf8").decode("utf8")
        xml_str = self.postprocess_model_xml(xml_str)
        tmp_dir = Path(tempfile.gettempdir()) / "robocasa365_mjcf_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        fd, new_xml_path = tempfile.mkstemp(prefix=f"{name}_", suffix=".xml", dir=str(tmp_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(xml_str)
            MujocoXMLObject.__init__(
                self,
                fname=new_xml_path,
                name=name,
                joints=[dict(type="free", damping="0.0005")],
                obj_type="all",
                duplicate_collision_geoms=False,
                scale=scale_arr,
            )
        finally:
            if os.path.exists(new_xml_path):
                os.remove(new_xml_path)

    MJCFObject.__init__ = patched_init
    setattr(MJCFObject, "_lawam_tmp_write_patch", True)


def _patch_kitchen_episode_meta(ep_meta: dict[str, Any]) -> None:
    from robocasa.environments.kitchen.kitchen import Kitchen

    Kitchen._lawam_episode_meta = dict(ep_meta)
    if getattr(Kitchen, "_lawam_ep_meta_patch", False):
        return
    orig_load_model = Kitchen._load_model

    def load_model_with_episode_meta(self):
        meta = getattr(Kitchen, "_lawam_episode_meta", None)
        if isinstance(meta, dict) and meta:
            self._ep_meta = dict(meta)
            # Older robocasa-murp cannot resolve some serialized fixture names
            # during model construction. Keep a copy for diagnostics, but let
            # this local env build its fixture registry normally.
            self._ep_meta.pop("fixture_refs", None)
            self._ep_meta.pop("clutter_mode", None)
        return orig_load_model(self)

    Kitchen._load_model = load_model_with_episode_meta
    setattr(Kitchen, "_lawam_ep_meta_patch", True)


def _patch_robot_init_qpos_length() -> None:
    from robosuite.robots.robot import Robot

    if getattr(Robot, "_lawam_init_qpos_patch", False):
        return
    orig_reset = Robot.reset

    def reset_with_truncated_init_qpos(self, deterministic=False):
        ref_len = len(getattr(self, "_ref_joint_pos_indexes", []) or [])
        init_qpos = getattr(self, "init_qpos", None)
        restore = None
        if init_qpos is not None and ref_len > 0 and len(init_qpos) != ref_len:
            restore = init_qpos
            self.init_qpos = tuple(init_qpos[:ref_len])
        try:
            return orig_reset(self, deterministic=deterministic)
        finally:
            if restore is not None:
                self.init_qpos = restore

    Robot.reset = reset_with_truncated_init_qpos
    setattr(Robot, "_lawam_init_qpos_patch", True)


def _patch_robot_right_base_site() -> None:
    from robosuite.robots.robot import Robot

    if getattr(Robot, "_lawam_right_base_site_patch", False):
        return

    def _arm_base_site(robot, arm: str) -> str:
        if arm != "right":
            return f"robot0_{arm}_center"
        names = set(robot.sim.model.site_names)
        return "robot0_right_base_center" if "robot0_right_base_center" in names else "robot0_right_center"

    def _transform_from_arm_base(robot, arm: str, *, use_site_eef: bool) -> np.ndarray:
        site_name = _arm_base_site(robot, arm)
        arm_base_T_world = np.eye(4)
        arm_base_T_world[:3, -1] = robot.sim.data.get_site_xpos(site_name)
        arm_base_T_world[:3, :3] = robot.sim.data.get_site_xmat(site_name)
        ee_T_world = np.eye(4)
        if use_site_eef:
            ee_T_world[:3, -1] = robot.sim.data.get_site_xpos(f"gripper0_{arm}_grip_site")
            ee_T_world[:3, :3] = robot.sim.data.get_site_xmat(f"gripper0_{arm}_grip_site")
        else:
            ee_T_world[:3, -1] = robot.sim.data.get_body_xpos(f"robot0_{arm}_hand")
            ee_T_world[:3, :3] = robot.sim.data.get_body_xmat(f"robot0_{arm}_hand")
        return np.linalg.inv(arm_base_T_world) @ ee_T_world

    def get_ee_T_arm_base(self, arm):
        return _transform_from_arm_base(self, arm, use_site_eef=False)

    def get_ee_site_T_arm_base(self, arm):
        return _transform_from_arm_base(self, arm, use_site_eef=True)

    Robot.get_ee_T_arm_base = get_ee_T_arm_base
    Robot.get_ee_site_T_arm_base = get_ee_site_T_arm_base
    setattr(Robot, "_lawam_right_base_site_patch", True)


def _patch_robot_arm_base_observable() -> None:
    from robosuite.robots.robot import Robot
    from robosuite.utils.observables import sensor

    if getattr(Robot, "_lawam_arm_base_observable_patch", False):
        return
    orig_create_arm_sensors = Robot._create_arm_sensors

    def create_arm_sensors_with_site_fallback(self, arm, modality):
        sensors, names = orig_create_arm_sensors(self, arm, modality)
        if arm != "right":
            return sensors, names

        @sensor(modality=modality)
        def arm_base_center_T_world_pose_mat(obs_cache):
            site_name = "robot0_right_base_center"
            if site_name not in set(self.sim.model.site_names):
                site_name = "robot0_right_center"
            arm_base_center_T_world = np.eye(4)
            arm_base_center_T_world[:3, -1] = self.sim.data.get_site_xpos(site_name)
            arm_base_center_T_world[:3, :3] = self.sim.data.get_site_xmat(site_name)
            return arm_base_center_T_world.flatten()

        for idx, name in enumerate(names):
            if str(name).endswith("arm_base_center_T_world_pose_mat"):
                sensors[idx] = arm_base_center_T_world_pose_mat

        @sensor(modality=modality)
        def gripper_keypoint_pose(obs_cache):
            gripper_key = "robot0_right_hand"
            gripper = getattr(self.robot_model, "grippers", {}).get(gripper_key, None)
            site_names = list(getattr(gripper, "sites", []) or []) if gripper is not None else []
            positions = []
            for site_name in site_names:
                try:
                    site_id = self.sim.model.site_name2id(site_name)
                    positions.append(np.asarray(self.sim.data.site_xpos[site_id], dtype=np.float64).copy())
                except Exception:
                    continue
            if not positions:
                return np.zeros(0, dtype=np.float64)
            return np.concatenate(positions)

        for idx, name in enumerate(names):
            if str(name).endswith("gripper_keypoint_pose"):
                sensors[idx] = gripper_keypoint_pose
        return sensors, names

    Robot._create_arm_sensors = create_arm_sensors_with_site_fallback
    setattr(Robot, "_lawam_arm_base_observable_patch", True)


def _patch_kitchen_reset_internal() -> None:
    from robocasa.environments.kitchen.kitchen import Kitchen
    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

    if getattr(Kitchen, "_lawam_reset_internal_patch", False):
        return

    def reset_internal_without_settle(self):
        ManipulationEnv._reset_internal(self)
        if not self.deterministic_reset and self.placement_initializer is not None:
            for obj_pos, obj_quat, obj in self.object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    Kitchen._reset_internal = reset_internal_without_settle
    setattr(Kitchen, "_lawam_reset_internal_patch", True)


def _patch_kitchen_bbox_observables() -> None:
    from robocasa.environments.kitchen.kitchen import Kitchen

    if getattr(Kitchen, "_lawam_bbox_observable_patch", False):
        return
    orig_create_obj_sensors = Kitchen._create_obj_sensors

    def create_obj_sensors_without_camera_bboxes(self, obj_name, modality="object"):
        sensors, names = orig_create_obj_sensors(self, obj_name, modality=modality)
        filtered = [
            (sensor_fn, name)
            for sensor_fn, name in zip(sensors, names)
            if "_bbox_in_" not in str(name)
        ]
        if not filtered:
            return [], []
        out_sensors, out_names = zip(*filtered)
        return list(out_sensors), list(out_names)

    Kitchen._create_obj_sensors = create_obj_sensors_without_camera_bboxes
    setattr(Kitchen, "_lawam_bbox_observable_patch", True)


def _patch_task_id_mapping_for_episode_xml() -> None:
    from robosuite.models.tasks.task import Task, get_subtree_geom_ids_by_group

    if getattr(Task, "_lawam_episode_xml_id_mapping_patch", False):
        return

    def _geom_ids(sim: Any, names: Any) -> list[int]:
        if isinstance(names, str):
            names = [names]
        out = []
        for name in list(names or []):
            try:
                out.append(int(sim.model.geom_name2id(name)))
            except Exception:
                continue
        return out

    def _site_ids(sim: Any, names: Any) -> list[int]:
        if isinstance(names, str):
            names = [names]
        out = []
        for name in list(names or []):
            try:
                out.append(int(sim.model.site_name2id(name)))
            except Exception:
                continue
        return out

    def generate_id_mappings_tolerant(self, sim):
        self._instances_to_ids = {}
        self._geom_ids_to_instances = {}
        self._site_ids_to_instances = {}
        self._classes_to_ids = {}
        self._geom_ids_to_classes = {}
        self._site_ids_to_classes = {}

        models = [model for model in self.mujoco_objects]
        for robot in self.mujoco_robots:
            models += [robot] + robot.models

        worldbody = self.mujoco_arena.root.find("worldbody")
        exclude_bodies = {"table", "left_eef_target", "right_eef_target"}
        if worldbody is not None:
            models.extend(
                body.attrib.get("name")
                for body in worldbody.findall("body")
                if body.attrib.get("name") not in exclude_bodies
            )

        for model in models:
            if model is None:
                continue
            if isinstance(model, str):
                body_name = model
                try:
                    body_id = sim.model.body_name2id(body_name)
                except Exception:
                    continue
                inst, cls = body_name, body_name
                id_groups = [get_subtree_geom_ids_by_group(sim.model, body_id, 1), []]
            else:
                cls = str(type(model)).split("'")[1].split(".")[-1]
                inst = model.name
                id_groups = [
                    _geom_ids(sim, list(model.visual_geoms or []) + list(model.contact_geoms or [])),
                    _site_ids(sim, list(model.sites or [])),
                ]

            if inst in self._instances_to_ids:
                continue
            self._instances_to_ids[inst] = {}
            if cls not in self._classes_to_ids:
                self._classes_to_ids[cls] = {"geom": [], "site": []}

            for ids, group_type, ids_to_inst, ids_to_cls in (
                (id_groups[0], "geom", self._geom_ids_to_instances, self._geom_ids_to_classes),
                (id_groups[1], "site", self._site_ids_to_instances, self._site_ids_to_classes),
            ):
                self._instances_to_ids[inst][group_type] = ids
                self._classes_to_ids[cls][group_type] += ids
                for idn in ids:
                    if idn in ids_to_inst:
                        continue
                    ids_to_inst[idn] = inst
                    ids_to_cls[idn] = cls

    Task.generate_id_mappings = generate_id_mappings_tolerant
    setattr(Task, "_lawam_episode_xml_id_mapping_patch", True)


def _patch_missing_model_visual_sites() -> None:
    from robocasa.environments.kitchen.kitchen import Kitchen
    from robosuite.models.base import MujocoModel

    if getattr(MujocoModel, "_lawam_missing_sites_visibility_patch", False):
        return
    orig_set_sites_visibility = MujocoModel.set_sites_visibility
    orig_kitchen_visualize = Kitchen.visualize

    def set_sites_visibility_tolerant(self, sim, visible):
        try:
            return orig_set_sites_visibility(self, sim=sim, visible=visible)
        except Exception:
            for site_name in list(getattr(self, "sites", []) or []):
                try:
                    site_id = sim.model.site_name2id(site_name)
                except Exception:
                    continue
                alpha = sim.model.site_rgba[site_id][3]
                if (visible and alpha < 0) or ((not visible) and alpha > 0):
                    sim.model.site_rgba[site_id][3] = -alpha
            return None

    def kitchen_visualize_tolerant(self, vis_settings):
        try:
            return orig_kitchen_visualize(self, vis_settings=vis_settings)
        except Exception:
            for obj in getattr(self.model, "mujoco_objects", []) or []:
                try:
                    obj.set_sites_visibility(sim=self.sim, visible=vis_settings["env"])
                except Exception:
                    pass
            for robot in getattr(self, "robots", []) or []:
                for geom_name in list(getattr(robot.robot_model, "visual_geoms", []) or []):
                    try:
                        rgba = self.sim.model.geom_rgba[self.sim.model.geom_name2id(geom_name)]
                    except Exception:
                        continue
                    rgba[-1] = 0.10 if getattr(self, "translucent_robot", False) else 1.0
            return None

    MujocoModel.set_sites_visibility = set_sites_visibility_tolerant
    Kitchen.visualize = kitchen_visualize_tolerant
    setattr(MujocoModel, "_lawam_missing_sites_visibility_patch", True)


def _patch_kitchen_update_state_for_episode_xml() -> None:
    from robocasa.environments.kitchen.kitchen import Kitchen
    from robosuite.environments.base import MujocoEnv

    if getattr(Kitchen, "_lawam_episode_xml_update_state_patch", False):
        return

    def update_state_tolerant(self):
        MujocoEnv.update_state(self)
        for fixture in list(getattr(self, "fixtures", {}).values()):
            try:
                fixture.update_state(self)
            except Exception:
                continue

    Kitchen.update_state = update_state_tolerant
    setattr(Kitchen, "_lawam_episode_xml_update_state_patch", True)


def _patch_hybrid_mobile_base_joint_position_dims() -> None:
    from robosuite.controllers.composite.composite_controller import HybridMobileBase

    if getattr(HybridMobileBase, "_lawam_joint_position_dim_patch", False):
        return

    def _fit_1d(value: Any, dim: int) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.size == dim and arr.ndim == 1:
            return arr.copy()
        flat = arr.reshape(-1)
        if flat.size == 1:
            return np.full(dim, flat[0], dtype=arr.dtype)
        return flat[:dim].copy()

    def _fit_position_limits(value: Any, dim: int) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim == 2 and arr.shape[0] == 2:
            return arr[:, :dim].copy()
        return arr

    def _normalize_joint_position_controller(controller: Any) -> int:
        if getattr(controller, "name", None) != "JOINT_POSITION":
            return 0
        qpos_index = getattr(controller, "qpos_index", None)
        qpos_dim = len(qpos_index or [])
        if qpos_dim <= 0:
            return 0
        for attr in (
            "input_max",
            "input_min",
            "output_max",
            "output_min",
            "kp",
            "kd",
            "kp_min",
            "kp_max",
            "damping_ratio_min",
            "damping_ratio_max",
        ):
            fitted = _fit_1d(getattr(controller, attr, None), qpos_dim)
            if fitted is not None:
                setattr(controller, attr, fitted)
        controller.position_limits = _fit_position_limits(getattr(controller, "position_limits", None), qpos_dim)
        controller.action_scale = None
        controller.action_input_transform = None
        controller.action_output_transform = None
        return qpos_dim

    def set_goal_with_joint_dim_fallback(self, all_action):
        for part_name, controller in self.part_controllers.items():
            start_idx, end_idx = self._action_split_indexes[part_name]
            action = np.asarray(all_action[start_idx:end_idx])
            if part_name in self.grippers.keys():
                action = self.grippers[part_name].format_action(action)
            else:
                qpos_dim = _normalize_joint_position_controller(controller)
                if qpos_dim > 0 and len(action) != qpos_dim:
                    action = action[:qpos_dim]

            if part_name in self.arms and hasattr(controller, "set_goal_update_mode"):
                goal_update_mode = "desired" if all_action[-1] > 0 else "achieved"
                controller.set_goal_update_mode(goal_update_mode)

            controller.set_goal(action)

    HybridMobileBase.set_goal = set_goal_with_joint_dim_fallback
    setattr(HybridMobileBase, "_lawam_joint_position_dim_patch", True)


def _load_episode_meta_for_env(ep_meta_path: Path) -> dict[str, Any]:
    import robocasa

    ep_meta = json.loads(ep_meta_path.read_text(encoding="utf-8"))
    ep_meta["_lawam_fixture_refs"] = dict(ep_meta.get("fixture_refs", {}) or {})
    ep_meta.pop("fixture_refs", None)
    ep_meta.pop("clutter_mode", None)
    asset_objects_root = Path(robocasa.models.assets_root) / "objects"
    for cfg in ep_meta.get("object_cfgs", []):
        # The final state is restored from episode XML/state. During the
        # throwaway env construction, object placements from RoboCasa365 can
        # reference fixture names that do not exist in this older robocasa-murp.
        cfg.pop("placement", None)
        cfg.pop("reset_region", None)
        info = cfg.get("info", None)
        if not isinstance(info, dict):
            continue
        mjcf_path = str(info.get("mjcf_path", ""))
        marker = "objects/"
        if marker in mjcf_path:
            rel = mjcf_path.split(marker, 1)[1]
            info["mjcf_path"] = str(asset_objects_root / rel)
    return ep_meta


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


def _load_episode_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"].item()))
    return {
        "metadata": metadata,
        "policy_action": np.asarray(data["policy_action_order"], dtype=np.float32),
        "policy_state": np.asarray(data["policy_state_order"], dtype=np.float32),
        "prompt": str(data["prompt"].item()),
        "timestamps": np.asarray(data["timestamps"], dtype=np.float32),
    }


def _load_dataset_env_meta(dataset_root: Path) -> dict[str, Any]:
    meta_path = dataset_root / "extras" / "dataset_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing RoboCasa365 dataset metadata: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_episode_states(episode: dict[str, Any]) -> tuple[str, np.ndarray]:
    states_path = Path(episode["metadata"]["states_npz"])
    model_path = Path(episode["metadata"]["model_xml_gz"])
    states = np.load(states_path)["states"]
    model_xml = _rewrite_model_xml_asset_paths(_read_model_xml(model_path))
    return model_xml, states


def _read_model_xml(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.read()


def _rewrite_model_xml_asset_paths(model_xml: str) -> str:
    import robocasa
    import robosuite

    robocasa_assets = Path(robocasa.models.assets_root)
    robosuite_assets = Path(robosuite.models.assets_root)

    def replace_file(match: re.Match[str]) -> str:
        original = match.group(1)
        try:
            if Path(original).exists():
                return f'file="{original}"'
        except OSError:
            pass
        markers = (
            ("robocasa/models/assets/", robocasa_assets),
            ("robosuite/models/assets/", robosuite_assets),
        )
        for marker, root in markers:
            if marker in original:
                rel = original.split(marker, 1)[1]
                return f'file="{root / rel}"'
        # RoboCasa object XMLs often only contain the suffix "objects/...".
        if "objects/" in original:
            rel = original.split("objects/", 1)[1]
            return f'file="{robocasa_assets / "objects" / rel}"'
        return f'file="{original}"'

    return re.sub(r'file="([^"]+)"', replace_file, model_xml)


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
    return image[::-1].copy()


def _render_image(env: Any, camera_name: str, height: int, width: int) -> np.ndarray | None:
    try:
        return np.asarray(env.sim.render(height=height, width=width, camera_name=camera_name), dtype=np.uint8)[::-1].copy()
    except Exception:
        return None


def _camera_frame(
    env: Any,
    obs: dict[str, Any],
    camera_name: str,
    height: int,
    width: int,
) -> np.ndarray | None:
    image = _obs_image(obs, f"{camera_name}_image")
    if image is not None:
        return image
    return _render_image(env, camera_name, height, width)


def _policy_action_to_env_action(action: np.ndarray, env: Any) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != 12:
        raise ValueError(f"Expected Panda policy action dim 12, got {action.shape[0]}.")
    env_dim = int(env.action_dim)
    if env_dim == 12:
        return action.astype(np.float32, copy=True)
    robot = env.robots[0]
    controller = getattr(robot, "composite_controller", None)
    if controller is not None and hasattr(controller, "create_action_vector"):
        def _fit_part(part_name: str, values: np.ndarray) -> np.ndarray:
            split = getattr(controller, "_action_split_indexes", {}).get(part_name, None)
            if split is None:
                return np.asarray(values, dtype=np.float32)
            target_dim = int(split[1] - split[0])
            out = np.zeros(target_dim, dtype=np.float32)
            src = np.asarray(values, dtype=np.float32).reshape(-1)
            out[: min(target_dim, src.shape[0])] = src[: min(target_dim, src.shape[0])]
            return out

        action_dict = {
            "right": _fit_part("right", action[0:6]),
            "right_gripper": _fit_part("right_gripper", action[6:7]),
            "base": _fit_part("base", action[7:11]),
            "base_mode": float(action[11]),
        }
        full_action = np.asarray(controller.create_action_vector(action_dict), dtype=np.float32)
        if full_action.shape[0] == env_dim:
            return full_action
    if env_dim > 12:
        full_action = np.zeros(env_dim, dtype=np.float32)
        full_action[:12] = action
        return full_action
    raise ValueError(f"Cannot adapt Panda policy action dim 12 to env action_dim={env_dim}.")


def _get_policy_example(
    env: Any,
    obs: dict[str, Any],
    *,
    prompt: str,
    state: np.ndarray,
    camera_height: int,
    camera_width: int,
) -> dict[str, Any]:
    right = _camera_frame(
        env,
        obs,
        POLICY_TO_SIM_CAMERA["robot0_agentview_right"],
        camera_height,
        camera_width,
    )
    wrist = _camera_frame(
        env,
        obs,
        POLICY_TO_SIM_CAMERA["robot0_eye_in_hand"],
        camera_height,
        camera_width,
    )
    left = _camera_frame(
        env,
        obs,
        POLICY_TO_SIM_CAMERA["robot0_agentview_left"],
        camera_height,
        camera_width,
    )
    if left is None and right is not None:
        # Some RoboCasa365 XMLs only carry robotview + wrist cameras. Keep the
        # policy input shape stable; the summary records the camera mapping.
        left = right.copy()
    missing = [
        name
        for name, value in (
            ("robot0_agentview_right/robot0_robotview", right),
            ("robot0_eye_in_hand/robot0_eye_in_hand", wrist),
            ("robot0_agentview_left/robot0_frontview", left),
        )
        if value is None
    ]
    if missing:
        raise KeyError(f"Missing required camera frames for policy example: {missing}.")
    return {
        "lang": str(prompt),
        "primary_image": [right, left],
        "wrist_image": [wrist],
        "wm_primary_image": right,
        "state": np.asarray(state, dtype=np.float32),
        "embodiment_id": 4,
        "action_hz": 20.0,
    }


def _make_env(args: argparse.Namespace, dataset_env_meta: dict[str, Any]):
    import robocasa  # noqa: F401
    import robosuite

    env_args = dataset_env_meta.get("env_args", {})
    env_kwargs = copy.deepcopy(env_args.get("env_kwargs", {}))
    if not env_kwargs:
        env_kwargs = copy.deepcopy(dataset_env_meta.get("env_info", {}))
    if not env_kwargs:
        raise ValueError("RoboCasa365 dataset_meta.json does not contain env_kwargs.")

    # RoboCasa365 metadata uses the 1.0 task name. This local robocasa-murp fork
    # still registers the older class name for the same task.
    env_name = str(env_args.get("env_name") or env_kwargs.get("env_name") or "")
    if env_name == "PickPlaceSinkToCounter":
        env_name = "PnPSinkToCounter"
    # RoboCasa365 1.0 metadata includes this field, while the local
    # robocasa-murp Kitchen constructor predates it.
    env_kwargs.pop("clutter_mode", None)
    env_kwargs.update(
        env_name=env_name,
        camera_names=VIDEO_CAMERA_NAMES,
        camera_widths=int(args.camera_width),
        camera_heights=int(args.camera_height),
        has_renderer=False,
        renderer="mjviewer",
        has_offscreen_renderer=(not bool(args.no_video)) or str(args.action_source) == "policy",
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_depths=False,
        seed=int(args.seed),
        randomize_cameras=False,
        translucent_robot=False,
    )
    env_kwargs.setdefault("control_freq", 20)
    if args.layout_id is not None:
        env_kwargs["layout_ids"] = [int(args.layout_id)]
    if args.style_id is not None:
        env_kwargs["style_ids"] = [int(args.style_id)]
    return robosuite.make(**env_kwargs)


def _reset_to_episode(env: Any, episode: dict[str, Any]) -> None:
    import robocasa.scripts.playback_utils as P

    model_xml, states = _load_episode_states(episode)
    P.reset_to(env, {"model": model_xml, "states": states[0]})


def _check_success(env: Any) -> bool:
    try:
        return bool(env._check_success())
    except Exception:
        return False


def _success_parts(env: Any) -> dict[str, Any]:
    try:
        import robocasa.utils.object_utils as OU

        parts = {
            "obj_in_receptacle": bool(OU.check_obj_in_receptacle(env, "obj", "container")),
            "container_on_counter": bool(env.check_contact(env.objects["container"], env.counter)),
            "gripper_obj_far": bool(OU.gripper_obj_far(env)),
        }
        for obj_name in ("obj", "container"):
            try:
                body_id = env.obj_body_id[obj_name]
                parts[f"{obj_name}_pos"] = np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).tolist()
            except Exception:
                pass
        try:
            parts["eef_pos"] = np.asarray(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]], dtype=np.float64).tolist()
        except Exception:
            pass
        return parts
    except Exception as exc:
        return {"error": repr(exc)}


def _write_outputs(
    output_dir: Path,
    *,
    frames: dict[str, list[np.ndarray]],
    actions: list[np.ndarray],
    diagnostics: list[dict[str, Any]],
    summary: dict[str, Any],
    fps: int,
) -> dict[str, str]:
    import imageio.v2 as imageio

    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, imgs in frames.items():
        if not imgs:
            continue
        path = output_dir / f"{name}.mp4"
        imageio.mimsave(path, imgs, fps=int(fps))
        written[f"{name}_video"] = str(path)
    action_path = output_dir / "actions.npy"
    np.save(action_path, np.asarray(actions, dtype=np.float32))
    written["actions"] = str(action_path)
    diagnostics_path = output_dir / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    written["diagnostics"] = str(diagnostics_path)
    summary_path = output_dir / "summary.json"
    payload = dict(summary)
    payload["outputs"] = written
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    written["summary_json"] = str(summary_path)
    return written


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--episode_index", type=int, required=True)
    parser.add_argument("--episode_npz", type=Path, required=True)
    parser.add_argument("--action_source", choices=("gt", "policy", "states"), default="gt")
    parser.add_argument("--policy_host", type=str, default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=6135)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--camera_height", type=int, default=256)
    parser.add_argument("--camera_width", type=int, default=256)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--layout_id", type=int, default=None)
    parser.add_argument("--style_id", type=int, default=None)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument(
        "--state_start_idx",
        type=int,
        default=0,
        help="Only used with --action_source states. Starts state replay from this saved simulator-state index.",
    )
    parser.add_argument(
        "--policy_state_source",
        choices=("dataset",),
        default="dataset",
        help="Use exported LeRobot state for policy input. Env-observation state is intentionally not enabled until GT replay is validated.",
    )
    return parser


def main() -> None:
    _patch_numba_cache()
    _patch_robocasa_object_sites()
    args = build_argparser().parse_args()
    import robocasa  # noqa: F401

    _patch_right_base_site_fallback()
    _patch_mjcf_object_tmp_writes()
    _patch_robot_init_qpos_length()
    _patch_robot_right_base_site()
    _patch_robot_arm_base_observable()
    _patch_kitchen_reset_internal()
    _patch_kitchen_bbox_observables()
    _patch_task_id_mapping_for_episode_xml()
    _patch_missing_model_visual_sites()
    _patch_kitchen_update_state_for_episode_xml()
    _patch_hybrid_mobile_base_joint_position_dims()

    episode = _load_episode_npz(args.episode_npz.expanduser().resolve())
    if int(episode["metadata"]["episode_index"]) != int(args.episode_index):
        raise ValueError(
            f"episode_npz index={episode['metadata']['episode_index']} does not match --episode_index={args.episode_index}."
        )
    if Path(episode["metadata"]["dataset_root"]).resolve() != args.dataset_root.expanduser().resolve():
        raise ValueError("episode_npz was exported from a different dataset_root.")

    dataset_env_meta = _load_dataset_env_meta(args.dataset_root.expanduser().resolve())
    ep_meta = _load_episode_meta_for_env(Path(episode["metadata"]["ep_meta_json"]))
    _patch_kitchen_episode_meta(ep_meta)
    env = _make_env(args, dataset_env_meta)
    model_xml, saved_states = _load_episode_states(episode)
    _reset_to_episode(env, episode)
    obs = env._get_observations(force_update=True)
    env_action_dim = int(env.action_dim)
    env_name = str(getattr(env, "name", ""))
    policy_action = np.asarray(episode["policy_action"], dtype=np.float32)
    policy_state = np.asarray(episode["policy_state"], dtype=np.float32)
    prompt = str(episode["prompt"])
    frames: dict[str, list[np.ndarray]] = {camera_name: [] for camera_name in VIDEO_CAMERA_NAMES}
    actions: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []

    policy_meta: dict[str, Any] | None = None
    if args.action_source == "policy":
        meta_response = _send_request(args.policy_host, args.policy_port, {"type": "meta"})
        if not meta_response.get("ok", False):
            raise RuntimeError(f"Failed to query policy metadata: {meta_response}")
        policy_meta = dict(meta_response["data"])

    max_steps = min(int(args.steps), int(policy_action.shape[0]))
    try:
        for step in range(max_steps):
            for cam_name in frames:
                image = None
                if not bool(args.no_video):
                    image = _camera_frame(env, obs, cam_name, int(args.camera_height), int(args.camera_width))
                if image is not None:
                    frames[cam_name].append(image)

            if args.action_source == "states":
                import robocasa.scripts.playback_utils as P

                state_idx = min(int(args.state_start_idx) + step, int(saved_states.shape[0]) - 1)
                P.reset_to(env, {"states": saved_states[state_idx]})
                obs = env._get_observations(force_update=True)
                action = np.zeros(int(env.action_dim), dtype=np.float32)
                action_policy_order = policy_action[min(step, int(policy_action.shape[0]) - 1)]
            elif args.action_source == "gt":
                action_policy_order = policy_action[step]
                action = _policy_action_to_env_action(action_policy_order, env)
                obs, _, _, _ = env.step(action)
            else:
                example = _get_policy_example(
                    env,
                    obs,
                    prompt=prompt,
                    state=policy_state[min(step, int(policy_state.shape[0]) - 1)],
                    camera_height=int(args.camera_height),
                    camera_width=int(args.camera_width),
                )
                response = _send_request(
                    args.policy_host,
                    args.policy_port,
                    {"type": "predict_action", "examples": [example]},
                )
                if not response.get("ok", False):
                    raise RuntimeError(f"Failed to query policy bridge: {response}")
                chunk = np.asarray(response["data"]["raw_actions"], dtype=np.float32)
                if chunk.ndim == 3:
                    chunk = chunk[0]
                action_policy_order = chunk[0]
                action = _policy_action_to_env_action(action_policy_order, env)
                obs, _, _, _ = env.step(action)
            actions.append(action)
            diagnostics.append(
                {
                    "step": int(step),
                    "state_idx": int(min(int(args.state_start_idx) + step, int(saved_states.shape[0]) - 1)),
                    "success": _check_success(env),
                    "success_parts": _success_parts(env),
                    "action_order": POLICY_ACTION_ORDER,
                    "action_l2": float(np.linalg.norm(action)),
                    "action_min": float(np.min(action)),
                    "action_max": float(np.max(action)),
                }
            )
            if diagnostics[-1]["success"]:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    success = bool(diagnostics[-1]["success"]) if diagnostics else False
    summary: dict[str, Any] = {
        "success": success,
        "steps": len(actions),
        "action_source": str(args.action_source),
        "episode_index": int(args.episode_index),
        "prompt": prompt,
        "policy_action_order": POLICY_ACTION_ORDER,
        "env_action_dim": env_action_dim,
        "env_name": env_name,
        "dataset_env_name": str(dataset_env_meta.get("env_args", {}).get("env_name", "")),
        "policy_to_sim_camera": POLICY_TO_SIM_CAMERA,
        "policy_meta": policy_meta,
        "note": "If action_source=gt fails, fix env/action replay before interpreting policy rollout.",
    }
    written = _write_outputs(
        args.output_dir.expanduser().resolve(),
        frames=frames,
        actions=actions,
        diagnostics=diagnostics,
        summary=summary,
        fps=int(args.video_fps),
    )
    summary["outputs"] = written
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
