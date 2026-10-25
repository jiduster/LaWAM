"""Convert user-accepted DexJoCo pose candidates to the interaction v2 sidecar."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from starVLA.interaction_data import audit_store, read_dataset
from starVLA.interaction_geometry import (
    ENTITY_NAMES,
    FEATURE_NAMES,
    SCHEMA,
    InteractionStore,
    validate_geometry,
)

TASKS = {"hanoi": "bimanual_hanoi", "microwave": "bimanual_microwave_cook"}
OBJECTS = {
    "bimanual_hanoi": ["hanoi_disk_large", "hanoi_disk_medium", "hanoi_disk_small", "hanoi_base"],
    "bimanual_microwave_cook": ["hot_dog", "microwave_object", "door"],
}
CLASSES = {
    "bimanual_hanoi": "PandaBimanualHanoiGymEnv",
    "bimanual_microwave_cook": "PandaBimanualMicrowaveCookGymEnv",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix(q):
    q = np.asarray(q)
    if not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-4):
        raise ValueError("Candidate quaternions are not finite unit wxyz values")
    return Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3))


def convert(task, world, state, model, joint=None):
    n = len(state)
    world = np.asarray(world, dtype=np.float32).reshape(n, len(OBJECTS[task]), 7)
    if not np.isfinite(world).all():
        raise ValueError("Candidate world pose contains nonfinite values")
    rot = matrix(world[..., 3:])
    bodies = {name: (world[:, i, :3], rot[:, i]) for i, name in enumerate(OBJECTS[task])}
    values, masks = [], []

    def add(pos, ori, angle=None, extent=None):
        value = np.concatenate(
            [pos, ori[..., :, 0], ori[..., :, 1], pos - state[:, 7:10], pos - state[:, :3],
             np.zeros((n, 1), np.float32) if angle is None else np.asarray(angle).reshape(n, 1),
             np.zeros((n, 1), np.float32) if extent is None else np.full((n, 1), extent, np.float32)],
            axis=-1)
        valid = np.ones((n, 17), dtype=bool)
        valid[:, 15], valid[:, 16] = angle is not None, extent is not None
        values.append(value)
        masks.append(valid)

    def fixed(kind, name):
        node = getattr(model, kind)(name)
        parent = model.body(int(node.bodyid[0])).name
        pos, ori = bodies[parent]
        return pos + np.einsum("nij,j->ni", ori, node.pos), ori @ matrix(node.quat)

    if task == "bimanual_hanoi":
        add(*bodies["hanoi_base"])
        for size in ("large", "medium", "small"):
            extent = max(np.max(np.abs(model.geom(f"hanoi_disk_{size}_collision_{s}").pos[:2])
                                + model.geom(f"hanoi_disk_{size}_collision_{s}").size[:2])
                         for s in ("pos_x", "neg_x", "pos_y", "neg_y"))
            add(*bodies[f"hanoi_disk_{size}"], extent=extent)
        for post in "abc":
            name = f"hanoi_post_{post}_collision"
            add(*fixed("geom", name), extent=float(model.geom(name).size[0]))
    else:
        if joint is None or np.asarray(joint).shape != (n, 1) or not np.isfinite(joint).all():
            raise ValueError("Microwave candidate is missing finite microjoint positions")
        add(*bodies["hot_dog"])
        add(*bodies["microwave_object"])
        position, _ = fixed("site", "door_p1")
        add(position, bodies["door"][1], angle=joint)
        for name in ("door_handle", "start_button"):
            add(*fixed("geom", name), angle=joint if name == "door_handle" else None,
                extent=float(np.max(model.geom(name).size)))
        origin, _ = fixed("site", "int_p0")
        edges = [fixed("site", "int_p" + axis)[0] - origin for axis in "xyz"]
        add(origin + sum(edges) / 2, bodies["microwave_object"][1],
            extent=max(float(np.linalg.norm(edge, axis=-1).max() / 2) for edge in edges))
    value, valid = np.stack(values, 1).astype(np.float32), np.stack(masks, 1)
    validate_geometry(value, valid, task)
    return value, valid


def check_model(model, geometry):
    for kind, key, name_key in (("site", "reference_sites", "site_name"), ("geom", "reference_geoms", "name")):
        for record in geometry.get(key, []):
            node = getattr(model, kind)(record[name_key])
            if model.body(int(node.bodyid[0])).name != record["parent_body"]:
                raise ValueError("Candidate task geometry uses a different parent body")
            if not np.allclose(node.pos, record["local_position_m"], atol=1e-8, rtol=0):
                raise ValueError("Candidate task geometry position differs from current XML")
            if not np.allclose(matrix(node.quat), matrix(record["local_quaternion_wxyz"]), atol=1e-8, rtol=0):
                raise ValueError("Candidate task geometry orientation differs from current XML")


def run(args):
    if not args.accept_reconstructed or not args.acceptance_note.strip():
        raise ValueError("Pass --accept-reconstructed and a nonempty --acceptance-note")
    task = TASKS.get(args.task, args.task)
    source = args.source_root.expanduser().resolve() / task
    dataset = args.data_root.expanduser().resolve()
    dataset = dataset if dataset.name == task else dataset / task
    meta_path = source / "pose_sidecars/manifest.json"
    meta = json.loads(meta_path.read_text())
    if meta["task"] != task or meta["object_names"] != OBJECTS[task]:
        raise ValueError("Candidate task/object order does not match")
    if meta.get("label_source") != "mujoco_raw_restore_reconstruction":
        raise ValueError("Unexpected candidate provenance")
    parquet = sorted((dataset / "data").rglob("*.parquet"))
    if len(parquet) != 1 or digest(parquet[0]) != meta["source_parquet_sha256"]:
        raise ValueError("Dataset Parquet differs from the candidate source; use the original dataset")
    fps, rows = read_dataset(dataset)
    target = args.output.expanduser().resolve() / task
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    from scripts.export_interaction_replay import PhysicsOnlyViewer
    module = importlib.import_module("dexjoco.sim.envs.panda_" + task + "_env")
    original = module.MujocoRenderer
    module.MujocoRenderer = PhysicsOnlyViewer
    env = None
    try:
        env = getattr(module, CLASSES[task])(image_obs=False, randomize=False, randomize_dynamics=False)
        check_model(env._model, json.loads((source / "task_geometry.json").read_text()))
        (target / "episodes").mkdir(parents=True)
        entries, accepted, excluded = {}, [], {}
        for ep, row in rows.items():
            path = source / "pose_sidecars" / f"episode_{ep:06d}.npz"
            audit = json.loads((source / "audit" / f"episode_{ep:06d}.json").read_text())
            with np.load(path, allow_pickle=False) as z:
                n = len(row["state"])
                if int(z["episode_index"]) != ep or z["object_names"].tolist() != OBJECTS[task]:
                    raise ValueError(f"Candidate episode/object mismatch: {path}")
                if not np.array_equal(z["frame_index"], row["frame_index"]):
                    raise ValueError(f"Candidate frame mismatch: {path}")
                if not np.array_equal(z["source_raw_frame_index"], row["frame_index"] + audit["trim"]):
                    raise ValueError(f"Candidate raw-frame mapping mismatch: {path}")
                if z["object_pose_available"].shape != (n,) or not z["object_pose_available"].all():
                    excluded[str(ep)] = "missing_current_pose"
                    continue
                if z["object_pose_available"].dtype != np.bool_ or not z["object_present"].all():
                    raise ValueError(f"Invalid candidate availability/presence: {path}")
                if z["object_root_pose_world"].shape != (n, len(OBJECTS[task]) * 7):
                    raise ValueError(f"Candidate world pose shape mismatch: {path}")
                value, valid = convert(task, z["object_root_pose_world"], row["state"], env._model,
                                       z["object_joint_positions"] if task.endswith("microwave_cook") else None)
                np.savez_compressed(target / "episodes" / f"episode-{ep:06d}.npz",
                                    interaction_geometry=value, interaction_valid=valid,
                                    state=row["state"], action=row["action"], timestamp=row["timestamp"],
                                    frame_index=row["frame_index"], source_raw_frame_index=z["source_raw_frame_index"])
                entries[str(ep)] = dict(length=n, valid=True, source_strict_valid=bool(z["object_pose_valid"].all()),
                                        source_sha256=digest(path), source_audit=audit)
                accepted.append(ep)
        if len(accepted) < 2:
            raise ValueError("Need at least two complete accepted episodes for train/validation")
        manifest = dict(schema=SCHEMA, task=task, entities=ENTITY_NAMES[task], features=FEATURE_NAMES,
                        coordinate_frame="world", fps=fps, episodes=entries,
                        label_source="mujoco_raw_restore_reconstruction",
                        acceptance_policy="user_accepted_reconstruction",
                        acceptance_note=args.acceptance_note,
                        caveat="Reconstructed labels accepted for experiments; not independently measured GT.",
                        accepted_episodes=accepted, excluded_episodes=excluded,
                        relative_hand_source="original dataset tcp_pose right xyz[0:3], left xyz[7:10]",
                        source_geometry_sha256=digest(source / "task_geometry.json"),
                        source_manifest_sha256=digest(meta_path))
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (target / "accepted_episodes.json").write_text(json.dumps(accepted) + "\n")
        audit_store(InteractionStore(target), task, rows, fps, accepted)
        print(f"{task}: accepted={len(accepted)}, excluded={len(excluded)}")
        print(target)
    finally:
        if env is not None:
            env.close()
        module.MujocoRenderer = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(TASKS) + tuple(ENTITY_NAMES))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept-reconstructed", action="store_true")
    parser.add_argument("--acceptance-note", required=True)
    run(parser.parse_args())
