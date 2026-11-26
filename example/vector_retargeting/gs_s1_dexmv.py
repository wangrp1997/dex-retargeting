"""Replay the DexMV mustard-bottle example on a GS S1 and score free-object physics."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from time import perf_counter

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from gs_s1_filter import LandmarkFilter
from gs_s1_mujoco import ContactStepper, JointStops, hand_scene_xml, load_hand
from gs_s1_rviz import (FINGER_JOINTS, MANO_TO_GS_S1, OBJECTIVES, SingleHandDetector,
                        build_constrained, build_solver, get_package_share_directory)
from single_hand_detector import OPERATOR2MANO_RIGHT


FPS = 25  # DexMV RelocationDemonstration.collect_time_step
DIGITS = ("th", "ff", "mf", "rf", "lf")
LIFT_M, HOLD_S, PLACE_M = .05, .5, .05
# Source: dexmv-sim/examples/visualize_retargeting.py, hand camera to object camera.
CAM_R = Rotation.from_matrix([[-.0178, -.7087, -.7053], [.7386, .4662, -.487],
                              [.6739, -.5296, .5151]]).as_matrix()
CAM_T = np.array([.4976, .3935, .3154])


def load_sequence(root):
    seq = root / "examples/retargeting_source/relocate_mustard_example_seq"
    ids = sorted(int(p.stem.split("_")[-1]) for p in (seq / "hand_pose").glob("joints_*.npy"))
    if len(ids) < 21 or not np.all(np.diff(ids) == 1):
        raise ValueError("Expected a continuous DexMV hand sequence")
    # Follow the example's ten-frame tracking warm-up, and match by ID.
    ids = ids[10:]
    hands = np.array([np.load(seq / f"hand_pose/joints_{i}.npy") for i in ids])
    # The official object files contain a pickled dictionary of YCB ID -> [R | t].
    obj = np.array([np.load(seq / f"object_pose/{i}.npy", allow_pickle=True).item()["5"] for i in ids])
    if not (np.isfinite(hands).all() and np.isfinite(obj).all()):
        raise ValueError("Missing hand or bottle observations")
    rotations = Rotation.from_matrix(obj[:, :, :3]).as_matrix()
    # This sample ships the inter-camera transform but no measured table frame.
    # Estimate up from the bottle's upright axis during the initial stationary interval.
    z = rotations[:10, :, 2].mean(axis=0)
    z /= np.linalg.norm(z)
    x = rotations[0, :, 0] - z * (z @ rotations[0, :, 0])
    x /= np.linalg.norm(x)
    axes = np.column_stack((x, np.cross(z, x), z))
    bottle = (obj[:, :, 3] - obj[0, :, 3]) @ axes
    orient = axes.T @ rotations
    world = (hands @ CAM_R.T + CAM_T - obj[0, :, 3]) @ axes
    palm = np.stack([SingleHandDetector.estimate_frame_from_hand_points(p)
                     @ OPERATOR2MANO_RIGHT @ MANO_TO_GS_S1 for p in world])
    local = (world - world[:, :1]) @ palm
    np.testing.assert_allclose(local @ palm.transpose(0, 2, 1) + world[:, :1], world, atol=1e-7)
    return ids, world, local, palm, bottle, orient


def make_scene(args, mesh_dir, control, ref_xyz):
    root = hand_scene_xml(args.assets, mesh_dir, args.hand)
    world = root.find("worldbody")
    base = world.find("body[@name='fabase']")
    ET.SubElement(base, "freejoint", name="hand_base")
    # Exclude the mounting fixture from environmental contacts. The palm and
    # all finger surfaces interact with the table, bottle and one another.
    for body in base.iter("body"):
        for geom in body.findall("geom"):
            active = body.get("name") not in ("fabase", "falink1", "pmbase")
            geom.set("contype", "1" if active else "0")
            geom.set("conaffinity", str((3 if control else 7) if active else 0))
    floor = world.find("geom[@name='floor_collision']")
    # The operator approaches from outside the support surface, initially
    # below its height. Bound the tabletop to the demonstrated workspace.
    # DexMV YCBRelocate uses a 1.2 x 1.2 m tabletop. Estimate the near
    # edge on the approach side, since its absolute pose is not distributed.
    center = [ref_xyz[:, 0].min()-.075+.6, .5*(ref_xyz[:, 1].min()+ref_xyz[:, 1].max())]
    floor.set("type", "box")
    floor.set("pos", f"{center[0]} {center[1]} -.025")
    floor.set("size", ".6 .6 .025")
    floor.set("contype", "2")
    floor.set("conaffinity", "5")
    floor.set("friction", "1 .5 .01")
    ET.SubElement(world, "geom", name="room_floor", type="plane", pos="0 0 -.75",
                  size="1 1 .05", contype="2", conaffinity="4")
    obj = ET.SubElement(world, "body", name="mustard")
    ET.SubElement(obj, "freejoint", name="bottle_free")
    folder = args.dexmv_root / "hand_imitation/env/models/assets/ycb/collision/006_mustard_bottle"
    meshes = sorted(folder.glob("*.stl"))
    if len(meshes) != 2:
        raise ValueError(f"Expected the two DexMV collision meshes in {folder}")
    for i, path in enumerate(meshes):
        name = f"bottle_{i}"
        ET.SubElement(root.find("asset"), "mesh", name=name, file=str(path.resolve()))
        ET.SubElement(obj, "geom", type="mesh", mesh=name, name=name, density="1000",
                      contype="4", conaffinity="2" if control else "3", condim="4",
                      friction="1 .005 .0001", margin=".003", rgba=".95 .63 .12 1")
    model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.geom_margin[model.geom_contype == 1] = .001
    return model


def bottle_vertices(model, data, body):
    # MuJoCo recenters mesh assets at compile time. Transform the compiled
    # vertices back into the free body's frame before measuring table clearance.
    parts = []
    for g in np.flatnonzero(model.geom_bodyid == body):
        mesh = model.geom_dataid[g]
        start = model.mesh_vertadr[mesh]
        verts = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        parts.append((world - data.xpos[body]) @ data.xmat[body].reshape(3, 3))
    return np.concatenate(parts)


def solve_commands(args, method, local, mesh_dir):
    solver, cad = build_solver(True, args.hand, objective=method)
    retarget = build_constrained(solver, load_hand(args.assets, mesh_dir, args.hand), method, args)
    filt = LandmarkFilter("one_euro")
    commands, times = [], []
    for i, p in enumerate(local):
        target = filt.update(p, i / FPS)
        t = perf_counter()
        commands.append(retarget.solve(target, 1 / FPS).copy())
        times.append((perf_counter() - t) * 1000)
    return np.array(commands), solver, cad, retarget.held, np.median(times)


def longest_run(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return int(np.max(edges[1::2] - edges[::2], initial=0))


def grasp_metrics(clearance, hand_contact, table_contact, ref_clearance, final_error):
    supported = (np.asarray(clearance) >= LIFT_M) & hand_contact & ~np.asarray(table_contact)
    hold_seconds = max(longest_run(supported) - 1, 0) / FPS
    minimum_samples = int(np.ceil(HOLD_S * FPS)) + 1
    windows = np.convolve(supported.astype(int), np.ones(minimum_samples), "full")[:len(supported)]
    acquired = np.flatnonzero(windows >= minimum_samples)
    dropped = None
    if len(acquired):
        # Three consecutive unsupported observations while the demonstration
        # still calls for lift distinguish premature loss from release at the goal.
        lost = (np.asarray(ref_clearance) >= LIFT_M) & (~np.asarray(hand_contact) | table_contact)
        dropped = longest_run(lost[acquired[0]:]) >= 3
    return dict(lift_success=bool(supported.any()), stable_grasp=bool(len(acquired)),
                longest_supported_lift_s=hold_seconds, early_drop=dropped,
                relocate_success=bool(len(acquired) and not dropped and table_contact[-1] and final_error <= PLACE_M))


def contact_metrics(model, data, bottle, floor):
    hand_depth = obj_depth = 0.
    supports = set()
    on_table = False
    force = np.empty(6)
    for i, c in enumerate(data.contact):
        ga, gb = int(c.geom1), int(c.geom2)
        a, b = model.geom_bodyid[[ga, gb]]
        depth = max(-c.dist, 0.)
        if bottle not in (a, b):
            if a != 0 and b != 0:
                hand_depth = max(hand_depth, depth)
            continue
        other = gb if a == bottle else ga
        if model.geom_bodyid[other] != 0:
            obj_depth = max(obj_depth, depth)
        if c.efc_address < 0:
            continue
        mj.mj_contactForce(model, data, i, force)
        if force[0] <= 1e-5:
            continue
        if other == floor:
            on_table = True
        elif model.geom_bodyid[other] != 0:
            supports.add(model.body(model.geom_bodyid[other]).name)
    return hand_depth, obj_depth, on_table, supports


def mean_axis_error(a, b):
    norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate distal vector")
    return np.degrees(np.arccos(np.clip((a * b).sum(axis=-1) / norm, -1., 1.)))


class PassiveCoupling:
    """Project each ideal 1:1 linkage onto its single mechanical degree of freedom."""

    def __init__(self, model):
        self.model = model
        children = [model.joint(f + "J1").id for f in ("FF", "MF", "RF", "LF")]
        parents = [model.joint(f + "J2").id for f in ("FF", "MF", "RF", "LF")]
        self.cq, self.pq = model.jnt_qposadr[children], model.jnt_qposadr[parents]
        self.cv, self.pv = model.jnt_dofadr[children], model.jnt_dofadr[parents]
        self.low = np.maximum(model.jnt_range[children, 0], model.jnt_range[parents, 0])
        self.high = np.minimum(model.jnt_range[children, 1], model.jnt_range[parents, 1])
        self.mass = np.empty((model.nv, model.nv))
        self.max_correction = 0.

    def residual(self, data):
        return np.max(np.abs(data.qpos[self.cq] - data.qpos[self.pq]))

    def apply(self, data):
        # The pair's mass block determines the common speed and preserves
        # its combined momentum before a mechanical stop is reached.
        mj.mj_fullM(self.model, self.mass, data.qM)
        cross = self.mass[self.pv, self.cv]
        total = self.mass[self.pv, self.pv] + 2 * cross + self.mass[self.cv, self.cv]
        weight = (self.mass[self.cv, self.cv] + cross) / total
        residual = data.qpos[self.cq] - data.qpos[self.pq]
        self.max_correction = max(self.max_correction, np.max(np.abs(residual)))
        linked = np.clip(data.qpos[self.pq] + weight * residual, self.low, self.high)
        speed = data.qvel[self.pv] + weight * (data.qvel[self.cv] - data.qvel[self.pv])
        speed[((linked <= self.low) & (speed < 0)) | ((linked >= self.high) & (speed > 0))] = 0.
        data.qpos[self.cq] = data.qpos[self.pq] = linked
        data.qvel[self.cv] = data.qvel[self.pv] = speed


def run(args, method, seq, mesh_dir, hand, stop_guard, control=False):
    solved = solve_commands(args, method, seq[2], mesh_dir)
    return replay_commands(args, seq, mesh_dir, hand, stop_guard, *solved, control=control)


def settle_initial_bottle(model, data, floor):
    """Let the bottle come to rest under gravity with the hand held at frame zero."""
    # A mesh bottle on a box table needs multiple simultaneous contact points
    # to resolve flat support. Keep the same contact model throughout the lift.
    model.opt.enableflags |= mj.mjtEnableBit.mjENBL_MULTICCD
    joint = model.joint("bottle_free")
    q = joint.qposadr.item()
    v = joint.dofadr.item()
    body = model.jnt_bodyid[joint.id]
    initial = data.qpos.copy()
    obj_q = slice(q, q+7)
    obj_v = slice(v, v+6)
    # Use separate data so the recorded hand starts with its original state.
    warmup = mj.MjData(model)
    warmup.qpos[:] = initial
    warmup.ctrl[:] = data.ctrl
    for _ in range(round(1. / model.opt.timestep)):
        mj.mj_step(model, warmup)
        bodies = model.geom_bodyid[warmup.contact.geom]
        if np.any((bodies == body).any(axis=1) & (bodies != 0).all(axis=1)):
            raise ValueError("The hand touches the bottle during initial settling")
        bottle_pose = warmup.qpos[obj_q].copy()
        bottle_velocity = warmup.qvel[obj_v].copy()
        warmup.qpos[:] = initial
        warmup.qpos[obj_q] = bottle_pose
        warmup.qvel[:] = 0.
        warmup.qvel[obj_v] = bottle_velocity
    mj.mj_forward(model, warmup)
    _, _, on_table, _ = contact_metrics(model, warmup, body, floor)
    speed = np.linalg.norm(warmup.qvel[obj_v][:3])
    angular_speed = np.linalg.norm(warmup.qvel[obj_v][3:])
    if not on_table or not (speed < .001 and angular_speed < .02):
        raise RuntimeError(f"Bottle did not settle on the table: {speed:.5f} m/s, {angular_speed:.5f} rad/s")
    # Keep the annotated horizontal position and heading on the level table.
    # Settling determines the height, roll and pitch needed for support.
    initial_rot = Rotation.from_quat(np.roll(initial[q+3:q+7], -1)).as_matrix()
    resting_rot = warmup.xmat[body].reshape(3, 3)
    heading_delta = np.arctan2(initial_rot[1, 0], initial_rot[0, 0]) - np.arctan2(resting_rot[1, 0], resting_rot[0, 0])
    rotation = Rotation.from_rotvec([0., 0., heading_delta]) * Rotation.from_matrix(resting_rot)
    data.qpos[q+2] = warmup.qpos[q+2]
    data.qpos[q+3:q+7] = np.roll(rotation.as_quat(), 1)
    data.qvel[obj_v] = 0.
    mj.mj_forward(model, data)


def replay_commands(args, seq, mesh_dir, hand, stop_guard, commands, solver, cad, held_solver, solve_ms, control=False):
    ids, hands, local, palm_rots, object_positions, object_rots = seq
    model = make_scene(args, mesh_dir, control, object_positions)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    bottle = model.body("mustard").id
    palm = model.site("palm_frame").id
    floor = model.geom("floor_collision").id
    base = model.body("fabase").id
    verts = bottle_vertices(model, data, bottle)
    floor_z = data.geom_xpos[floor, 2] + model.geom_size[floor, 2]
    dz = floor_z - (verts @ object_rots[0].T)[:, 2].min() + .0001
    wrist = hands[:, 0].copy()
    wrist[:, 2] += dz
    ref_pos = object_positions.copy()
    ref_pos[:, 2] += dz
    ref_clear = ref_pos[:, 2] + np.min(verts @ object_rots[:, 2, :].T, axis=0) - floor_z
    local_base = data.xmat[base].reshape(3, 3)
    base_to_palm = local_base.T @ data.site_xmat[palm].reshape(3, 3)
    offset = local_base.T @ (data.site_xpos[palm] - data.xpos[base])
    mount_rot = Rotation.from_matrix(palm_rots @ base_to_palm.T)
    mount_xyz = wrist - mount_rot.apply(np.broadcast_to(offset, wrist.shape))
    root_q = model.joint("hand_base").qposadr.item()
    root_v = model.joint("hand_base").dofadr.item()
    obj_q = model.joint("bottle_free").qposadr.item()
    data.qpos[root_q:root_q+3] = mount_xyz[0]
    data.qpos[root_q+3:root_q+7] = np.roll(mount_rot[0].as_quat(), 1)
    data.qpos[obj_q:obj_q+3] = ref_pos[0]
    data.qpos[obj_q+3:obj_q+7] = np.roll(Rotation.from_matrix(object_rots[0]).as_quat(), 1)
    drive_names = [model.joint(j).name for j in model.actuator_trnid[:, 0]]
    assert set(drive_names) == set(FINGER_JOINTS) and len(drive_names) == 16
    pairs = {(model.joint(a).name, model.joint(b).name) for a,b,t in
             zip(model.eq_obj1id, model.eq_obj2id, model.eq_type) if t == mj.mjtEq.mjEQ_JOINT}
    assert pairs == {(f + "J1", f + "J2") for f in ("FF", "MF", "RF", "LF")}
    finger_names = [hand.joint(i).name for i in range(hand.njnt)]
    finger_q = [model.joint(name).qposadr.item() for name in finger_names]
    finger_v = [model.joint(name).dofadr.item() for name in finger_names]
    pin_idx = [solver.joint_names.index(n) for n in finger_names]
    drive_idx = [solver.joint_names.index(n) for n in drive_names]
    for child, parent in pairs:
        np.testing.assert_allclose(commands[:, solver.joint_names.index(child)],
                                   commands[:, solver.joint_names.index(parent)], atol=1e-8)
    stops = JointStops(model)
    # Start at the same observed timestamp in annotation and simulation. First
    # command was computed from neutral with the same one-frame speed limit.
    data.qpos[finger_q] = commands[0, pin_idx]
    data.ctrl[:] = commands[0, drive_idx]
    mj.mj_forward(model, data)
    np.testing.assert_allclose(data.site_xpos[palm], wrist[0], atol=1e-7)
    np.testing.assert_allclose(data.site_xmat[palm].reshape(3, 3), palm_rots[0], atol=1e-7)
    if getattr(args, "settle_bottle", False):
        settle_initial_bottle(model, data, floor)
    steps = round(1 / (FPS * model.opt.timestep))
    if not np.isclose(steps * model.opt.timestep, 1 / FPS):
        raise ValueError("DexMV frame times must align with simulation steps")
    grid = np.arange(steps*(len(ids)-1)+1) / steps
    interp_rot = Slerp(np.arange(len(ids)), mount_rot)(grid)
    quats = np.roll(interp_rot.as_quat(), 1, axis=1)
    translations = np.column_stack([np.interp(grid, np.arange(len(ids)), c) for c in mount_xyz.T])
    # Prescribe both pose and twist on the common six-axis mount. In MuJoCo
    # free-joint linear velocity is world-frame and angular velocity is body-frame.
    # Only the mount is prescribed; the hinges and bottle evolve under contact.
    mounts = np.column_stack([translations, quats])
    twists = np.column_stack([np.diff(translations,axis=0),
                              (interp_rot[:-1].inv()*interp_rot[1:]).as_rotvec()])/model.opt.timestep
    tips = [model.site(d + "tip_frame").id for d in DIGITS]
    distal = [model.body(d + "distal").id for d in DIGITS]
    tip_positions, directions, obj_pos, obj_rot, clearance, contacts = [], [], [], [], [], []
    on_table, penetration, cad_hits, wrist_err, angle, tip_contact, poses = [], [], [], [], [], [], []
    rejected = 0
    max_dynamic_depth = np.zeros(2)
    body_for_geom = model.geom_bodyid
    max_mimic_error = 0.
    coupling = PassiveCoupling(model)
    for i, frame_id in enumerate(ids):
        if i:
            data.ctrl[:] = commands[i, drive_idx]
            for step in range((i-1)*steps+1, i*steps+1):
                data.qvel[root_v:root_v+6] = twists[step-1]
                prev = data.qpos[finger_q].copy()
                mj.mj_step(model, data)
                # mj_step assembled contacts for the current dynamics step.
                # Track transient depth as well as frame-rate statistics.
                contact = data.contact
                penetrating = contact.dist < 0
                if np.any(penetrating):
                    bodies = body_for_geom[contact.geom[penetrating]]
                    internal = (bodies != 0).all(axis=1)
                    touches_object = (bodies == bottle).any(axis=1)
                    depths = -contact.dist[penetrating]
                    max_dynamic_depth = np.maximum(max_dynamic_depth, [
                        np.max(depths[internal & ~touches_object], initial=0.),
                        np.max(depths[internal & touches_object], initial=0.)])
                data.qpos[root_q:root_q+7] = mounts[step]
                stops.apply(data)
                coupling.apply(data)
                if not stop_guard.clear_pose(data.qpos[finger_q]):
                    data.qpos[finger_q] = prev
                    data.qvel[finger_v] = 0
                    data.qacc_warmstart[finger_v] = 0
                    rejected += 1
        mj.mj_forward(model, data)
        if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
            raise RuntimeError(f"Invalid simulation at DexMV frame {frame_id}")
        rot = data.site_xmat[palm].reshape(3, 3)
        tip_positions.append((data.site_xpos[tips] - data.site_xpos[palm]) @ rot)
        directions.append((data.site_xpos[tips] - data.xpos[distal]) @ rot)
        obj_pos.append(data.xpos[bottle].copy())
        obj_rot.append(data.xmat[bottle].reshape(3, 3).copy())
        clearance.append(float(obj_pos[-1][2] + (verts @ obj_rot[-1][2]).min() - floor_z))
        hand_depth, obj_depth, table_contact, support = contact_metrics(model, data, bottle, floor)
        contacts.append(bool(support))
        on_table.append(table_contact)
        tip_contact.append([f+"distal" in support for f in DIGITS])
        penetration.append([hand_depth, obj_depth])
        q = solver.optimizer.robot.q0.copy()
        q[pin_idx] = data.qpos[finger_q]
        cad_hits.append(bool(cad.first_intersection(q)))
        wrist_err.append(np.linalg.norm(wrist[i]-data.site_xpos[palm]))
        angle.append((Rotation.from_matrix(palm_rots[i]).inv()*Rotation.from_matrix(rot)).magnitude())
        poses.append(data.qpos.copy())
        max_mimic_error = max(max_mimic_error, coupling.residual(data))
    tracks = dict(frame=ids, qpos=poses, object_xyz=obj_pos, object_rotation=obj_rot,
                  reference_xyz=ref_pos, reference_rotation=object_rots, wrist_xyz=wrist,
                  wrist_rotation=palm_rots, reference_clearance_m=ref_clear,
                  clearance_m=clearance, table_contact=on_table, hand_contact=contacts,
                  distal_contact=tip_contact, penetration_m=penetration, cad_intersection=cad_hits,
                  tips_palm=tip_positions, distal_vectors=directions, human_palm=local)
    tip = np.array(tip_positions)
    human_tip = local[:, 4::4] - local[:, :1]
    tip_error = np.linalg.norm(tip-human_tip, axis=-1)*1000
    thumb_error = np.linalg.norm((tip[:, 1:]-tip[:, :1])-(human_tip[:, 1:]-human_tip[:, :1]), axis=-1)*1000
    direction = mean_axis_error(np.array(directions), local[:, 4::4]-local[:, 3::4])
    pos_error = np.linalg.norm(np.array(obj_pos)-ref_pos, axis=-1)
    rot_error = (Rotation.from_matrix(object_rots).inv()*Rotation.from_matrix(obj_rot)).magnitude()
    depth = np.array(penetration)*1000
    driven_q = np.array(poses)[:, [model.joint(n).qposadr.item() for n in drive_names]]
    result = dict(five_tip_wrist_mean_mm=tip_error.mean(), per_digit_tip_mm=dict(zip(DIGITS, tip_error.mean(axis=0))),
                  thumb_index_gap_mae_mm=1000*np.abs(np.linalg.norm(tip[:, 0]-tip[:, 1],axis=1)-
                                                       np.linalg.norm(human_tip[:, 0]-human_tip[:, 1],axis=1)).mean(),
                  four_tip_thumb_mean_mm=thumb_error.mean(), distal_direction_mean_deg=direction.mean(),
                  per_digit_direction_deg=dict(zip(DIGITS, direction.mean(axis=0))),
                  p95_max_joint_step_deg=np.percentile(np.degrees(np.abs(np.diff(driven_q,axis=0))).max(axis=1),95),
                  solver_holds=held_solver, rejected_self_collision_substeps=rejected, median_solve_ms=solve_ms,
                  max_wrist_tracking_mm=1000*max(wrist_err), max_wrist_tracking_deg=np.degrees(max(angle)),
                  self_cad_intersection_frames=sum(cad_hits),
                  sampled_self_penetration_frames=int(sum(depth[:, 0]>0)), sampled_self_max_mm=depth[:, 0].max(),
                  physics_self_peak_mm=max_dynamic_depth[0]*1000, physics_hand_object_peak_mm=max_dynamic_depth[1]*1000,
                  passive_mimic_max_residual_deg=np.degrees(max_mimic_error),
                  passive_mimic_largest_preprojection_step_deg=np.degrees(coupling.max_correction),
                  hand_object_penetration_frames=int(sum(depth[:, 1]>0)), hand_object_max_mm=depth[:, 1].max(),
                  hand_object_mean_depth_mm=depth[:, 1].mean(),
                  hand_object_contact_frames=sum(contacts),
                  per_digit_distal_contact_frames=dict(zip(DIGITS,np.sum(tip_contact,axis=0).tolist())),
                  reference_peak_bottom_clearance_mm=1000*ref_clear.max(),
                  reference_min_bottom_clearance_mm=1000*ref_clear.min(),
                  reference_continuous_lift_s=max(longest_run(ref_clear>=LIFT_M)-1,0)/FPS,
                  peak_bottom_clearance_mm=1000*max(clearance),
                  final_position_error_mm=1000*pos_error[-1],
                  object_position_mean_mm=1000*pos_error.mean(), object_rotation_mean_deg=np.degrees(rot_error.mean()),
                  pose_tracking_under_100mm_and_0p5rad=bool(pos_error.mean()<.1 and rot_error.mean()<.5),
                  bottle_mass_kg=model.body_mass[bottle])
    result.update(grasp_metrics(clearance, contacts, on_table, ref_clear, pos_error[-1]))
    return result, tracks


def main():
    ex = Path(__file__).resolve().parent
    proj = ex.parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dexmv-root", type=Path, default=proj/"dexmv-sim")
    p.add_argument("--assets", type=Path, default=proj/"tendon-spin/assets")
    p.add_argument("--wuji-root", type=Path, default=proj/"wuji-retargeting")
    p.add_argument("--tsinghua-root", type=Path, default=proj/"tsinghua-retargeting")
    p.add_argument("--objectives", nargs="+", choices=OBJECTIVES, default=OBJECTIVES)
    p.add_argument("--with-control", action="store_true", help="Also replay the first method without hand-bottle contact")
    p.add_argument("--settle-bottle", action="store_true", help="Let the free bottle rest on the table for 1 s before replay")
    p.add_argument("--output", type=Path, required=True, help="JSON report")
    p.add_argument("--tracks", type=Path, help="Write per-frame NPZ for inspection")
    args = p.parse_args()
    args.hand = "right"
    seq = load_sequence(args.dexmv_root)
    mesh = Path(get_package_share_directory("dexhand_description"))/"meshes/gs_s1"
    report = dict(sequence="relocate_mustard_example_seq", first_frame=seq[0][0], last_frame=seq[0][-1],
                  frames=len(seq[0]), fps=FPS, data_source="DexMV supplied 3D tracks",
                  table="1.2 m square; estimated from initial upright bottle, 0.1 mm initial clearance; near edge 75 mm outside object path",
                  physics_timestep_s=.0005, gravity_m_s2=9.81, wrist="prescribed common 6D motion with substep pose and velocity",
                  initial_bottle_settle_s=1. if args.settle_bottle else 0.,
                  finger_actuators=16, passive_joints=["FFJ1","MFJ1","RFJ1","LFJ1"],
                  lift_bottom_m=LIFT_M, continuous_hold_s=HOLD_S,
                  final_position_tolerance_m=PLACE_M, smoothing="One Euro", straight_hand_calibration=False,
                  metric_sources=["https://mingrui-yu.github.io/retargeting/",
                                  "https://computationalrobotics.seas.harvard.edu/GRASP/"],
                  methods={})
    tracks = {}
    jobs = [(n, False) for n in args.objectives]
    if args.with_control:
        jobs.append((args.objectives[0], True))
    for name, control in jobs:
        hand = load_hand(args.assets, mesh, args.hand)
        result, path = run(args, name, seq, mesh, hand, ContactStepper(hand), control)
        if control:
            if result["hand_object_contact_frames"] or result["lift_success"] or result["physics_hand_object_peak_mm"]:
                raise RuntimeError("No-contact negative control failed")
            report["no_contact_control"] = {"objective": name, **result}
        else:
            report["methods"][name] = result
        key = "no_contact_control" if control else name
        if args.tracks:
            tracks.update({f"{key}_{k}": v for k,v in path.items()})
        print(key, "grasp", result["stable_grasp"], "peak clearance mm", round(result["peak_bottom_clearance_mm"],1),flush=True)
    if args.tracks:
        np.savez_compressed(args.tracks, **tracks)
    args.output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
