from __future__ import annotations


RAW_ACTION_ORDER = [
    "base_motion_0",
    "base_motion_1",
    "base_motion_2",
    "base_motion_3",
    "control_mode",
    "eef_pos_x",
    "eef_pos_y",
    "eef_pos_z",
    "eef_rot_x",
    "eef_rot_y",
    "eef_rot_z",
    "gripper_close",
]

POLICY_ACTION_ORDER = [
    "eef_pos_x",
    "eef_pos_y",
    "eef_pos_z",
    "eef_rot_x",
    "eef_rot_y",
    "eef_rot_z",
    "gripper_close",
    "base_motion_0",
    "base_motion_1",
    "base_motion_2",
    "base_motion_3",
    "control_mode",
]

RAW_STATE_ORDER = [
    "base_pos_x",
    "base_pos_y",
    "base_pos_z",
    "base_quat_x",
    "base_quat_y",
    "base_quat_z",
    "base_quat_w",
    "eef_pos_rel_x",
    "eef_pos_rel_y",
    "eef_pos_rel_z",
    "eef_quat_rel_x",
    "eef_quat_rel_y",
    "eef_quat_rel_z",
    "eef_quat_rel_w",
    "gripper_qpos_0",
    "gripper_qpos_1",
]

POLICY_STATE_ORDER = [
    "eef_pos_rel_x",
    "eef_pos_rel_y",
    "eef_pos_rel_z",
    "eef_quat_rel_x",
    "eef_quat_rel_y",
    "eef_quat_rel_z",
    "eef_quat_rel_w",
    "gripper_qpos_0",
    "gripper_qpos_1",
    "base_pos_x",
    "base_pos_y",
    "base_pos_z",
    "base_quat_x",
    "base_quat_y",
    "base_quat_z",
    "base_quat_w",
]
