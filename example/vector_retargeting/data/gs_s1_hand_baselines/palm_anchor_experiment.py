"""Paired fixed-wrist experiment on the DexMV mustard clip. Run as a script."""
import argparse
import csv
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from gs_s1_dexmv import DIGITS, FPS
from gs_s1_filter import LandmarkFilter
from gs_s1_hand_baselines import angle_deg
from gs_s1_mujoco import load_hand
from gs_s1_optimization import ConstrainedRetargeter
from gs_s1_rviz import build_solver, get_package_share_directory
from measure_finger_shape import patch_normal, unit

HERE = Path(__file__).resolve().parent
# Central MANO volar vertices nearest the midpoint of the two distal flank
# contact anchors per digit in the neutral right-hand template (manotorch).
PAD_VERTEX = (763, 343, 455, 566, 683)
WEAK_WEIGHT = .1
LENGTH = .010
ANGLE = np.deg2rad(20.)
JOINT_MOTION_WEIGHT = .03
# Fixed relative weights for the optional phalanx objective, shared by all frames.
PHALANX_DIRECTION_WEIGHT = 4.
FLEXION_PLANE_WEIGHT = 32.
THUMB_PAD_WEIGHT = .2


def surface_targets(surface, track):
    np.testing.assert_array_equal(surface['frame'], track['frame'])
    xyz = surface['joints'] - surface['joints'][:, :1]
    u, _, vt = np.linalg.svd(xyz.transpose(0, 2, 1) @ track['raw_human_palm_m'])
    rot = u @ vt
    np.testing.assert_allclose(np.linalg.det(rot), 1., atol=1e-6)
    np.testing.assert_allclose(xyz @ rot, track['raw_human_palm_m'], atol=1e-6)
    verts = (surface['vertices'] - surface['joints'][:, :1]) @ rot
    faces = surface['faces']
    patches = [faces[(faces == v).any(axis=1)] for v in PAD_VERTEX]
    normal = np.stack([patch_normal(verts, p) for p in patches], axis=1)
    # A conservative lower bound on the closest points of each pair of modeled
    # surface patches. Every triangle lies inside its vertex bounding sphere.
    center = verts[:, PAD_VERTEX]
    radii = np.stack([np.linalg.norm(verts[:, np.unique(p)] - center[:, i:i+1], axis=-1).max(axis=1)
                      for i, p in enumerate(patches)], axis=1)
    lower = np.linalg.norm(center[:, 1:]-center[:, :1], axis=2)-radii[:, 1:]-radii[:, :1]
    return center, normal, lower


def cad_patches(mesh_dir, side='right'):
    dtype = [('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]
    centers, normals = [], []
    for finger in DIGITS:
        right_thumb = finger == 'th' and side == 'right'
        sign = -1 if right_thumb else 1
        path = mesh_dir / ('rh' if right_thumb else '') / (finger+'distal.STL')
        faces = np.fromfile(path, dtype=dtype, offset=84)
        a, b, c = faces['vertices'].astype(float).transpose(1, 0, 2)
        ab, ac = b-a, c-a
        det = ab[:, 0]*ac[:, 2] - ab[:, 2]*ac[:, 0]
        hits, directions = [], []
        for x in np.arange(-2, 3)*.001:
            for z in np.arange(16, 23)*.001:
                with np.errstate(divide='ignore', invalid='ignore'):
                    u = ((x-a[:, 0])*ac[:, 2] - (z-a[:, 2])*ac[:, 0])/det
                    v = (ab[:, 0]*(z-a[:, 2]) - ab[:, 2]*(x-a[:, 0]))/det
                    ids = np.flatnonzero((u>=0) & (v>=0) & (u+v<=1))
                ys = a[ids,1]+u[ids]*ab[ids,1]+v[ids]*ac[ids,1]
                pick = np.argmax(sign*ys)
                n = faces['normal'][ids[pick]]
                if sign*n[1] <= 0:
                    raise ValueError(f'Inward normal at {finger} pad')
                hits.append([x, ys[pick], z])
                directions.append(n)
        centers.append(np.mean(hits, axis=0))
        normals.append(unit(np.mean(directions, axis=0)))
    return np.array(centers), np.array(normals)


def bone_geometry(chain):
    segments = np.diff(chain, axis=-2)
    lengths = np.linalg.norm(segments, axis=-1, keepdims=True)
    bones = segments / lengths
    a, b = bones[..., :-1, :], bones[..., 1:, :]
    axis = np.cross(a, b)
    sine = np.linalg.norm(axis, axis=-1)
    bend = np.arctan2(sine, (a * b).sum(axis=-1))
    return bones, lengths, axis, sine, bend


def finger_shape(chain):
    bones, _, _, _, bend = bone_geometry(chain)
    return bones[..., 0, :], bend


class Hand:
    def __init__(self, solver, mesh_dir, side='right'):
        model = solver.geom.model
        self.model = model
        self.data = mj.MjData(model)
        self.mapping = solver.geom.mapping
        self.ids = np.array([[model.body(d+s).id for s in ('proximal','middle','distal','tip')] for d in DIGITS])
        mj.mj_kinematics(model, self.data)
        palm = model.body('palm').id
        self.wrist = self.data.xpos[palm].copy()
        self.axes = self.data.xmat[palm].reshape(3,3).copy()
        roots = (self.data.xpos[self.ids[[1, 4], 0]] - self.wrist) @ self.axes
        self.index_anchor = roots[0]
        self.palm_width = np.linalg.norm(roots[0] - roots[1])
        self.radial_axis = (roots[0] - roots[1]) / self.palm_width
        self.centers, self.normals = cad_patches(mesh_dir, side)
        fingers = ('FF', 'MF', 'RF', 'LF')
        drives = list(solver.opt.target_joint_names)
        self.spread_indices = [drives.index(f + 'J4') for f in fingers]
        spread = np.array([self.data.xaxis[model.joint(f + 'J4').id] for f in fingers]) @ self.axes
        flex = np.array([self.data.xaxis[model.joint(f + 'J3').id] for f in fingers]) @ self.axes
        # J4 rotates the J3 flexion axis. Rodrigues' formula gives its direction
        # in the palm frame for either hand, without dividing by projected bone length.
        parallel = spread * np.sum(spread * flex, axis=-1, keepdims=True)
        self.flexion_basis = np.stack((parallel, flex - parallel, np.cross(spread, flex)), axis=1)
        neutral_thumb = (self.data.xpos[self.ids[0]] - self.wrist) @ self.axes
        self.thumb_lengths = np.linalg.norm(np.diff(neutral_thumb, axis=0), axis=-1)
        self.thumb_zero_bends = bone_geometry(neutral_thumb)[4]
        self.thumb_flexion_indices = [drives.index(n) for n in ('THJ2', 'THJ1')]

    def thumb_bend_seed(self, chain, previous):
        """Initialize on the flexion branch beyond the CAD's tiny zero offset."""
        bends = bone_geometry(chain)[4][0]
        seed = previous.copy()
        seed[self.thumb_flexion_indices] = [bends[0] + self.thumb_zero_bends[0],
                                            bends[1] - self.thumb_zero_bends[1]]
        return seed

    def thumb_direction_target(self, chain):
        """Robot-length-normalized direction of the complete observed thumb."""
        bones = bone_geometry(chain)[0][0]
        return unit(np.sum(self.thumb_lengths[:, None] * bones, axis=0))

    def thumb_refinement_loss(self, chain, normal):
        """Track signed reach and pad orientation during the distal-gap refinement."""
        reach = thumb_span_in_palm_widths(chain[0, -1], chain[1, 0], chain[4, 0])
        axis = self.axes @ self.radial_axis
        index = self.wrist + self.axes @ self.index_anchor
        target = np.dot(index, axis) + self.palm_width * reach
        target_normal = self.axes @ normal
        distal, tip = self.ids[0, 2:]
        jac_pos = np.empty((3, self.model.nv))
        jac_rot = np.empty_like(jac_pos)
        weight = .15

        def loss(data, grad):
            # The gap objective has already evaluated kinematics for this pose.
            delta = np.dot(data.xpos[tip], axis) - target
            direction = data.xmat[distal].reshape(3, 3) @ self.normals[0]
            error = direction - target_normal
            if grad.size:
                # The tip is fixed to the distal body and shares its angular Jacobian.
                mj.mj_jacBody(self.model, data, jac_pos, jac_rot, tip)
                slope = delta / LENGTH**2 * (axis @ jac_pos)
                slope += np.cross(direction, error) @ jac_rot / ANGLE**2
                grad += 2 * weight * (slope @ self.mapping)
            return weight * ((delta / LENGTH)**2 + np.dot(error, error) / ANGLE**2)

        return loss

    def actuator_gradient(self, point_grad, pad_grad, normal_grad, pos, pads, normals):
        """Pull palm-frame derivatives back through the current fixed-wrist pose."""
        # A distal rotation moves the pad center and rotates its surface normal.
        torque = np.cross(pads - pos[:, 2], pad_grad) + np.cross(normals, normal_grad)
        point_grad[:, 2] += pad_grad
        force_world = point_grad @ self.axes.T
        torque_world = np.zeros_like(force_world)
        torque_world[:, 2] = torque @ self.axes.T
        mj.mj_comPos(self.model, self.data)
        joint_grad = np.zeros(self.model.nv)
        for body, force, moment in zip(self.ids.flat, force_world.reshape(-1, 3),
                                       torque_world.reshape(-1, 3)):
            # J_position.T @ force + J_rotation.T @ moment, at the body origin.
            mj.mj_applyFT(self.model, self.data, force, moment, self.data.xpos[body], body, joint_grad)
        return joint_grad @ self.mapping  # includes the four passive J1 joints

    def states(self, controls):
        # Perturb the independent actuators only. The same compiled coupling
        # matrix used for collision checking expands the four passive joints.
        controls = np.atleast_2d(controls)
        pos = np.empty((len(controls),5,4,3))
        rot = np.empty((len(controls),5,3,3))
        for i, q in enumerate(controls @ self.mapping.T):
            self.data.qpos[:] = q
            mj.mj_kinematics(self.model, self.data)
            pos[i] = self.data.xpos[self.ids]
            rot[i] = self.data.xmat[self.ids[:,2]].reshape(5,3,3)
        pos = (pos-self.wrist) @ self.axes
        offset = np.einsum('bfij,fj->bfi', rot, self.centers) @ self.axes
        normal = np.einsum('bfij,fj->bfi', rot, self.normals) @ self.axes
        return pos, pos[:,:,2]+offset, normal


def filtered(series):
    filt = LandmarkFilter('one_euro')
    return np.stack([filt.update(v, i/FPS).copy() for i,v in enumerate(series)])


def shape_point_gradient(bones, lengths, axis, sine, root_error, bend_error, direction_error=None):
    """Differentiate bone directions and unsigned bend angles with respect to chain points."""
    root = bones[:, 0]
    root_grad = 2 * root_error / (len(root) * ANGLE**2)
    bone_grad = np.zeros_like(bones)
    bone_grad[:, 0] = root_grad - root * (root * root_grad).sum(axis=-1, keepdims=True)
    if direction_error is not None:
        directions = bones[1:]
        tangents = direction_error - directions * (directions * direction_error).sum(axis=-1, keepdims=True)
        count = directions.shape[0] * directions.shape[1]
        bone_grad[1:] += (2 * PHALANX_DIRECTION_WEIGHT / (count * ANGLE**2)) * tangents

    a, b = bones[:, :-1], bones[:, 1:]
    sine = sine[..., None]
    # At collinearity the unsigned angle has a cusp. Its symmetric derivative
    # is zero, matching centered differences at the exactly straight pose.
    plane = np.divide(axis, sine, out=np.zeros_like(axis), where=sine > 64 * np.finfo(float).eps)
    bend_grad = (2 / (bend_error.size * ANGLE**2)) * bend_error[..., None]
    bone_grad[:, :-1] += bend_grad * np.cross(a, plane)
    bone_grad[:, 1:] += bend_grad * np.cross(plane, b)
    segment_grad = bone_grad / lengths
    point_grad = np.zeros((len(bones), bones.shape[1] + 1, 3))
    point_grad[:, :-1] -= segment_grad
    point_grad[:, 1:] += segment_grad
    return point_grad


def thumb_span_in_palm_widths(tip, index_mcp, little_mcp):
    """Signed projection beyond the index MCP along the little-to-index axis."""
    across = index_mcp - little_mcp
    return np.dot(tip - index_mcp, across) / np.dot(across, across)


def direction_objective(hand, chain, previous, *, gradient="finite_difference", thumb_base_only=False):
    """Fit each digit's directions and bends independently of human bone lengths.

    Robot roots and segment lengths determine the resulting fingertip positions.
    Cross-digit positions and surface gaps are omitted. Thumb flexion can be
    fitted independently of distal absolute directions for base-only refinement.
    """
    target_bones, _, _, _, target_bend = bone_geometry(chain)
    direction_weights = np.ones((5, 3, 1))
    if thumb_base_only:
        # Thumb flexion follows its intrinsic bend angles. Distal absolute
        # directions must not bend J1/J2 to compensate a changed base roll.
        direction_weights[0, 1:] = 0
    direction_count = direction_weights.sum()

    def losses(x, grad=None):
        pos, pads, normals = hand.states(x)
        bones, lengths, axis, sine, bend = bone_geometry(pos)
        direction_error = bones - target_bones
        bend_error = bend - target_bend
        terms = np.sum(direction_weights * (direction_error / ANGLE)**2, axis=(1, 2, 3)) / direction_count
        terms += np.mean((bend_error / ANGLE)**2, axis=(1, 2))
        terms += JOINT_MOTION_WEIGHT * np.mean(((x - previous) / ANGLE)**2, axis=1)
        if grad is not None:
            # Unit-vector derivatives live in the tangent plane of each bone.
            bones, lengths = bones[0], lengths[0]
            error = direction_error[0]
            tangents = error - bones * np.sum(bones * error, axis=-1, keepdims=True)
            segment_grad = 2 * direction_weights * tangents / (direction_count * ANGLE**2 * lengths)
            point_grad = shape_point_gradient(bones, lengths, axis[0], sine[0],
                                              np.zeros_like(bones[:, 0]), bend_error[0])
            point_grad[:, :-1] -= segment_grad
            point_grad[:, 1:] += segment_grad
            grad[:] = hand.actuator_gradient(point_grad, np.zeros_like(pads[0]),
                                             np.zeros_like(normals[0]), pos[0], pads[0], normals[0])
            grad += 2 * JOINT_MOTION_WEIGHT * (x[0] - previous) / (len(previous) * ANGLE**2)
        return terms

    if gradient == "analytic":
        def analytic(x, grad):
            return float(losses(x[None], grad if grad.size else None)[0])
        return analytic
    if gradient != "finite_difference":
        raise ValueError(f"Unknown shape gradient: {gradient}")
    n, eps = len(previous), 1e-5
    displacements = np.vstack((np.zeros((1, n)), eps * np.eye(n), -eps * np.eye(n)))

    def callback(x, grad):
        values = losses(x + displacements if grad.size else x[None])
        if grad.size:
            grad[:] = (values[1:n + 1] - values[n + 1:]) / (2 * eps)
        return float(values[0])

    return callback


def thumb_base_objective(hand, chain, previous, *, gradient="finite_difference", opponents=None):
    """Fit the thumb's overall ray; callers fix flexion and the other fingers.

    There is one continuous objective over the entire workspace. Palm crossing,
    prescribed inside/outside roll angles are absent. Fixed opposing pad centres
    define a smooth facing field, without a contact or crossing classification.
    """
    target = hand.thumb_direction_target(chain)

    def losses(x, grad=None):
        pos, pads, normals = hand.states(x)
        ray = pos[:, 0, -1] - pos[:, 0, 0]
        length = np.linalg.norm(ray, axis=-1, keepdims=True)
        direction = ray / length
        error = direction - target
        terms = np.sum((error / ANGLE)**2, axis=-1)
        terms += JOINT_MOTION_WEIGHT * np.mean(((x - previous) / ANGLE)**2, axis=1)
        if opponents is not None:
            delta = opponents[None] - pads[:, :1]
            distance_sq = np.sum(delta**2, axis=-1, keepdims=True) + LENGTH**2
            field = np.sum(delta / distance_sq, axis=1)
            field_length = np.sqrt(np.sum(field**2, axis=-1, keepdims=True) + 1e-12)
            field_direction = field / field_length
            pad_error = normals[:, 0] - field_direction
            terms += THUMB_PAD_WEIGHT * np.sum((pad_error / ANGLE)**2, axis=-1)
        if grad is not None:
            tangent = error[0] - direction[0] * np.dot(direction[0], error[0])
            force = 2 * tangent / (ANGLE**2 * length[0, 0])
            points = np.zeros_like(pos[0])
            points[0, 0], points[0, -1] = -force, force
            pad_grad, normal_grad = np.zeros_like(pads[0]), np.zeros_like(normals[0])
            if opponents is not None:
                vectors, square = delta[0], distance_sq[0, :, 0]
                slope = np.sum(-np.eye(3)[None] / square[:, None, None]
                               + 2 * vectors[:, :, None] * vectors[:, None, :] / square[:, None, None]**2, axis=0)
                u = field_direction[0]
                slope = (np.eye(3) - np.outer(u, u)) @ slope / field_length[0, 0]
                normal_grad[0] = 2 * THUMB_PAD_WEIGHT * pad_error[0] / ANGLE**2
                pad_grad[0] = -slope.T @ normal_grad[0]
            grad[:] = hand.actuator_gradient(points, pad_grad, normal_grad, pos[0], pads[0], normals[0])
            grad += 2 * JOINT_MOTION_WEIGHT * (x[0] - previous) / (len(previous) * ANGLE**2)
        return terms

    if gradient == "analytic":
        def analytic(x, grad):
            return float(losses(x[None], grad if grad.size else None)[0])
        return analytic
    if gradient != "finite_difference":
        raise ValueError(f"Unknown shape gradient: {gradient}")
    n, eps = len(previous), 1e-5
    displacements = np.vstack((np.zeros((1, n)), eps * np.eye(n), -eps * np.eye(n)))

    def callback(x, grad):
        values = losses(x + displacements if grad.size else x[None])
        if grad.size:
            grad[:] = (values[1:n + 1] - values[n + 1:]) / (2 * eps)
        return float(values[0])

    return callback


def objective(hand, chain, pad, normal, previous, weight, *, opposition=False, gradient="finite_difference",
              phalanx=False, thumb_span=False):
    target_bones, _, _, _, target_bend = bone_geometry(chain)
    target_root = target_bones[:, 0]
    if phalanx:
        # A matched proximal direction lies in the robot finger's flexion plane.
        # Its dot product with the J3 axis is a smooth signed residual, also at full flexion.
        offset, cosine, sine_coeff = np.einsum('fkc,fc->kf', hand.flexion_basis, target_root[1:])
    if thumb_span:
        # Signed reach beyond the index MCP, measured in palm widths. The fixed
        # robot palm anchors keep this term independent of long-finger spread.
        reach = thumb_span_in_palm_widths(chain[0, -1], chain[1, 0], chain[4, 0])
        radial_target = hand.palm_width * reach
    target_vectors = pad[1:] - pad[:1]
    # Nearer pads need finer relative accuracy. Use the same continuous rule
    # for all four pairs, bounded at zero gap by the length tolerance.
    pair_weights = np.ones(len(target_vectors))
    if opposition:
        palm_width_sq = np.sum((chain[1,0] - chain[4,0])**2)
        pair_weights = 1 + palm_width_sq / (LENGTH**2 + np.sum(target_vectors**2, axis=-1))
    # Distances and angles are normalized by 10 mm and 20 degrees, then each
    # term is averaged across the corresponding digits. No pinch projection.
    def losses(x, grad=None):
        pos, pads, normals = hand.states(x)
        bones, lengths, axis, sine, bend = bone_geometry(pos)
        root_error = bones[..., 0, :] - target_root
        bend_error = bend - target_bend
        pair_delta = pads[:, 1:] - pads[:, :1] - target_vectors
        pair_errors = np.sum((pair_delta / LENGTH)**2, axis=-1)
        terms = (pair_errors * pair_weights if opposition else pair_errors).mean(axis=1)
        terms += np.mean(np.sum((root_error/ANGLE)**2, axis=-1),axis=1)
        terms += np.mean((bend_error/ANGLE)**2, axis=(1,2))
        terms += np.sum(((normals[:,0]-normal[0])/ANGLE)**2, axis=-1)
        if opposition:
            pad_angles = np.sum(((normals[:,1:]-normal[1:])/ANGLE)**2, axis=-1)
            terms += (np.sqrt(pair_weights) * pad_angles).mean(axis=1)
        centroid_delta = pads.mean(1)-pad.mean(0)
        terms += weight*np.sum((centroid_delta/LENGTH)**2, axis=-1)
        terms += JOINT_MOTION_WEIGHT * np.mean(((x - previous) / ANGLE)**2, axis=1)
        if phalanx:
            spread = x[:, hand.spread_indices]
            plane_error = offset + cosine * np.cos(spread) + sine_coeff * np.sin(spread)
            direction_error = bones[:, 1:] - target_bones[1:]
            terms += FLEXION_PLANE_WEIGHT * np.mean((plane_error / ANGLE)**2, axis=1)
            terms += PHALANX_DIRECTION_WEIGHT * np.mean(np.sum((direction_error / ANGLE)**2, axis=-1), axis=(1, 2))
            # Retain thumb pad orientation when fitting the directions of the other digits.
            terms += np.sum(((normals[:, 0] - normal[0]) / ANGLE)**2, axis=-1)
        if thumb_span:
            radial_delta = (pos[:, 0, -1] - hand.index_anchor) @ hand.radial_axis - radial_target
            terms += (radial_delta / LENGTH)**2
        if grad is not None:
            # Analytical evaluations contain a single pose. hand.data is still at x[0].
            pos, pads, normals = pos[0], pads[0], normals[0]
            point_grad = shape_point_gradient(bones[0], lengths[0], axis[0], sine[0],
                                              root_error[0], bend_error[0], direction_error[0] if phalanx else None)
            if thumb_span:
                point_grad[0, -1] += 2 * radial_delta[0] * hand.radial_axis / LENGTH**2
            pair_grad = (2 / (len(target_vectors) * LENGTH**2)) * pair_weights[:, None] * pair_delta[0]
            pad_grad = np.vstack((-pair_grad.sum(axis=0), pair_grad))
            pad_grad += (2 * weight / (len(pads) * LENGTH**2)) * centroid_delta[0]
            normal_grad = np.zeros_like(normals)
            normal_grad[0] = (2 / ANGLE**2) * (normals[0] - normal[0])
            if phalanx:
                normal_grad[0] *= 2
            if opposition:
                normal_grad[1:] = (2 / (len(target_vectors) * ANGLE**2)) * (
                    np.sqrt(pair_weights)[:, None] * (normals[1:] - normal[1:]))
            grad[:] = hand.actuator_gradient(point_grad, pad_grad, normal_grad, pos, pads, normals)
            if phalanx:
                slope = sine_coeff * np.cos(spread[0]) - cosine * np.sin(spread[0])
                grad[hand.spread_indices] += 2 * FLEXION_PLANE_WEIGHT * plane_error[0] * slope / (len(slope) * ANGLE**2)
            grad += (2 * JOINT_MOTION_WEIGHT / (len(previous) * ANGLE**2)) * (x[0] - previous)
        return terms

    if gradient == "analytic":
        def analytic(x, grad):
            return float(losses(x[None, :], grad if grad.size else None)[0])
        return analytic
    if gradient != "finite_difference":
        raise ValueError(f"Unknown shape gradient: {gradient}")

    n = len(previous)
    eps = 1e-5
    displacements = np.vstack([np.zeros((1,n)), eps*np.eye(n), -eps*np.eye(n)])
    def callback(x, grad):
        values = losses(x+displacements if grad.size else x[None,:])
        if grad.size:
            grad[:] = (values[1:n+1]-values[n+1:])/(2*eps)
        return float(values[0])
    return callback


def run(args):
    track = np.load(HERE/'trajectories.npz')
    surface = np.load(args.surface)
    mesh_dir = Path(get_package_share_directory('dexhand_description'))/'meshes/gs_s1'
    mesh_targets, normals, gap_bound = surface_targets(surface,track)
    hands = track['raw_human_palm_m'][:,1:].reshape(-1,5,4,3)
    input_chain = track['filtered_human_palm_m'][:,1:].reshape(hands.shape)
    input_pad = filtered(mesh_targets)
    input_normal = unit(filtered(normals))
    human_root, human_bend = finger_shape(hands)
    human_vec = mesh_targets[:,1:]-mesh_targets[:,:1]
    ids = track['frame']
    archive = dict(frame=ids, joint_names=track['joint_names'])
    report = dict(sequence='DexMV mustard, right hand, frames 11-122 at 25 Hz',
                  input='paired methods share filtered MANO pads, normals and landmarks; DexPilot uses filtered landmarks only; score raw MANO',
                  weak_weight=WEAK_WEIGHT, length_unit_m=LENGTH, angle_unit_deg=np.degrees(ANGLE),
                  temporal_weight=.03, method='soft least squares, 4 inter-pad vectors, 5 proximal directions, 10 bone angles, thumb pad normal, previous command; optional single five-pad centroid',
                  min_human_pad_patch_gap_lower_bound_mm=(1000*gap_bound.min(0)).tolist(),
                  methods={})
    rows = []
    for name, weight in [('dexpilot',None),('no_anchor',0.),('weak_anchor',WEAK_WEIGHT)]:
        seq, _ = build_solver()
        solver = ConstrainedRetargeter(seq,load_hand(args.assets,mesh_dir,'right'))
        np.testing.assert_array_equal(seq.optimizer.target_joint_names, track['joint_names'])
        hand = Hand(solver,mesh_dir)
        if weight is None:
            q = track['dexpilot_active_qpos_rad']
        else:
            output = []
            for i in range(len(ids)):
                fn = objective(hand,input_chain[i],input_pad[i],input_normal[i],solver.last.copy(),weight)
                solver.solve_objective(fn,1/FPS)
                output.append(solver.last.copy())
            q = np.stack(output)
        assert np.isfinite(q).all() and np.all(q >= solver.limits[:,0]-1e-7) and np.all(q <= solver.limits[:,1]+1e-7)
        for cmd in q:
            assert solver.geom.feasible(cmd) and solver.palm.clear(solver.geom.data,solver.geom.clearance-1e-7)
            for d in ('FF','MF','RF','LF'):
                joint = solver.geom.model.joint
                assert solver.geom.data.qpos[joint(d+'J1').qposadr] == solver.geom.data.qpos[joint(d+'J2').qposadr]
        archive[name+'_active_qpos_rad'] = q
        pos,pad,norm = hand.states(q)
        root,bend = finger_shape(pos)
        shape_error = np.degrees(abs(bend-human_bend))
        diff = pad[:,1:]-pad[:,:1]
        vec_err = 1000*np.linalg.norm(diff-human_vec,axis=-1)
        gap_err = 1000*abs(np.linalg.norm(diff,axis=-1)-np.linalg.norm(human_vec,axis=-1))
        normal_err = angle_deg(norm[:,0],normals[:,0])
        centroid_err = 1000*np.linalg.norm(pad.mean(1)-mesh_targets.mean(1),axis=-1)
        tip_abs = 1000*np.linalg.norm(pos[:,:,-1]-hands[:,:,-1],axis=-1)
        if name=='dexpilot':
            with (HERE/'dexpilot_finger_shape.csv').open() as f:
                prior=[r for r in csv.DictReader(f) if r['digit']=='th']
            np.testing.assert_allclose(tip_abs[:,0], [float(r['tip_error_mm']) for r in prior],atol=.025)
            np.testing.assert_allclose(normal_err,[float(r['thumb_pad_angle_deg']) for r in prior],atol=.05)
        focus = list(ids).index(88)
        thumb_j1 = list(track['joint_names']).index('THJ1')
        report['methods'][name] = dict(bend_mae_deg=shape_error.mean(), thumb_distal_bend_mae_deg=shape_error[:,0,1].mean(),
             thumb_pad_normal_mean_deg=normal_err.mean(), thumb_pad_normal_median_deg=np.median(normal_err),
             relative_pad_vector_mean_mm=vec_err.mean(), relative_pad_distance_mae_mm=gap_err.mean(),
             thumb_index_pad_distance_mae_mm=gap_err[:,0].mean(), proximal_direction_mean_deg=angle_deg(root,human_root).mean(),
             pad_centroid_mean_mm=centroid_err.mean(), absolute_tip_mean_mm=tip_abs.mean(),
             held_frames=solver.held if weight is not None else json.loads((HERE/'scores.json').read_text())['methods']['dexpilot']['solver_held_frames'],
             checked_feasible_frames=len(q),
             thumb_j1_within_one_degree_of_lower=int(np.sum(q[:,thumb_j1] <= solver.limits[thumb_j1,0]+np.deg2rad(1))),
             thumb_j1_within_one_degree_of_upper=int(np.sum(q[:,thumb_j1] >= solver.limits[thumb_j1,1]-np.deg2rad(1))),
             p95_max_joint_step_deg=np.percentile(np.degrees(abs(np.diff(q,axis=0))).max(1),95),
             frame88=dict(thumb_distal_bend_deg=np.degrees(bend[focus,0,1]),thumb_pad_normal_deg=normal_err[focus],
                          ring_distal_bend_deg=np.degrees(bend[focus,3,1]),relative_pad_vector_mean_mm=vec_err[focus].mean()))
        for i,frame in enumerate(ids):
            for j,d in enumerate(DIGITS):
                rows.append(dict(method=name,frame=int(frame),digit=d,mano_mid_deg=np.degrees(human_bend[i,j,0]),
                          robot_mid_deg=np.degrees(bend[i,j,0]),mano_distal_deg=np.degrees(human_bend[i,j,1]),
                          robot_distal_deg=np.degrees(bend[i,j,1]),thumb_pad_normal_deg=normal_err[i] if not j else '',
                          thumb_to_digit_vec_err_mm=vec_err[i,j-1] if j else '',thumb_to_digit_gap_err_mm=gap_err[i,j-1] if j else '',
                          absolute_tip_mm=tip_abs[i,j]))
        print(name, json.dumps(report['methods'][name]), flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'per_frame.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.out/'trajectories.npz',**archive)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,default=Path.home()/'Projects/tendon-spin/assets')
    parser.add_argument('--surface',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=HERE/'palm_anchor_ablation')
    run(parser.parse_args())
