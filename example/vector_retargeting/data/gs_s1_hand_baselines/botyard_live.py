"""Causal MANO fitting and selectable Botyard GS S1 pose objectives."""
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from distal_closure import closure_weights, mano_surface_gaps
from fit_example_mano import ORDER, TIPS, distal_face_labels
from gs_s1_filter import LandmarkFilter
from gs_s1_mano_overlay import palm_frame
from gs_s1_optimization import ConstrainedRetargeter
from gs_s1_rviz import MANO_TO_GS_S1
from palm_anchor_experiment import Hand, direction_objective, objective, surface_targets, thumb_base_objective, unit
from single_hand_detector import OPERATOR2MANO_LEFT, OPERATOR2MANO_RIGHT, SingleHandDetector
from shape_priority_closure import ALGORITHM_VERSION, ProtectedSurfaceClosure, SOLVER_FAILURES
from sweep_closure import configure_closure


class OnlineMano:
    """Fit the pose of the local licensed MANO template to one RGB observation at a time."""

    def __init__(self, path, *, thumb_direction_fit=False):
        if not path.is_file():
            raise FileNotFoundError(f'Missing prepared MANO template {path}. See the Botyard camera instructions.')
        with np.load(path, allow_pickle=False) as model:
            self.template = model['v_template']
            self.rest = model['J_regressor'] @ self.template
            self.parents = model['kintree_table'][0].astype(int)
            self.weights = model['weights']
            self.directions = model['posedirs']
            self.faces = model['f']
            self.labels = distal_face_labels(self.faces, self.weights)
            self.mean = model['hands_mean'].astype(float)
        self.pose = self.mean.copy()
        self.keypoint_pose = self.mean.copy()
        self.tips = (self.template[TIPS], self.weights[TIPS], self.directions[TIPS])
        self.mesh = (self.template, self.weights, self.directions)
        self.perturbations = np.vstack((np.zeros(45), 1e-5 * np.eye(45)))
        self.scale_filter = LandmarkFilter('one_euro')
        self.last_time = None
        self.preview_mesh = None
        self.thumb_direction_fit = thumb_direction_fit

    def skin(self, poses, root, geometry):
        """Batched, wrist-centred MANO forward kinematics and linear blend skinning."""
        local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 15, 3, 3)
        batch = len(poses)
        rotations = np.empty((batch, 16, 3, 3))
        joints = np.zeros((batch, 16, 3))
        rotations[:, 0] = root
        for j, parent in enumerate(self.parents[1:], 1):
            joints[:, j] = joints[:, parent] + rotations[:, parent] @ (self.rest[j] - self.rest[parent])
            rotations[:, j] = rotations[:, parent] @ local[:, j-1]
        translations = joints - (rotations @ self.rest[..., None])[..., 0]
        transforms = np.concatenate((rotations, translations[..., None]), axis=-1).reshape(batch, 16, 12)
        template, weights, directions = geometry
        offsets = (local - np.eye(3)).reshape(batch, 135) @ directions.reshape(-1, 135).T
        posed = template + offsets.reshape(batch, -1, 3)
        blend = (weights @ transforms).reshape(batch, -1, 3, 4)
        vertices = (blend[..., :3] @ posed[..., None])[..., 0] + blend[..., 3]
        return joints, vertices

    def fit(self, observed, pixels, size, timestamp):
        # Match the four MANO MCP joints to the detected palm in camera coordinates.
        palm = self.rest[[1, 4, 10, 7]] - self.rest[0]
        measured = observed[[5, 9, 13, 17]]
        u, _, vt = np.linalg.svd(palm.T @ measured)
        correction = np.diag([1., 1., np.linalg.det(u @ vt)])
        root = (u @ correction @ vt).T
        scale = np.sum((palm @ root.T) * measured) / np.sum(palm**2)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Degenerate observed palm')
        scale = self.scale_filter.update(np.array([[scale]]), timestamp).item()
        # Estimate camera translation from all image rays and wrist-relative 3D points.
        focal = max(size)
        rays = (pixels - np.asarray(size) / 2) / focal
        matrix = np.tile(np.eye(2, 3), (len(rays), 1))
        matrix[:, 2] = -rays.ravel()
        translation = np.linalg.lstsq(matrix, (rays * observed[:, 2:] - observed[:, :2]).ravel(), rcond=None)[0]
        fresh = self.last_time is None or timestamp - self.last_time > .5
        previous = self.mean if fresh else (self.keypoint_pose if self.thumb_direction_fit else self.pose)

        def residual(poses):
            joints, tips = self.skin(poses, root, self.tips)
            fitted = scale * np.concatenate((joints, tips), axis=1)[:, ORDER]
            terms = [((fitted - observed) / .006).reshape(len(poses), -1)]
            if translation[2] > .08:
                camera = fitted + translation
                projected = camera[..., :2] / np.maximum(camera[..., 2:], .05)
                terms.append(((projected - rays) * (640. / 4.)).reshape(len(poses), -1))
            terms.extend((.8 * (poses - self.mean), 1.2 * (poses - previous)))
            return np.concatenate(terms, axis=1)

        def jacobian(pose):
            values = residual(pose + self.perturbations)
            return (values[1:] - values[0]).T / 1e-5

        result = least_squares(lambda pose: residual(pose[None])[0], previous, jac=jacobian,
                               loss='soft_l1', max_nfev=9 if fresh else 5)
        self.keypoint_pose = result.x.copy()
        previous_tip = self.mean[-3:] if fresh else self.pose[-3:]
        self.pose = result.x.copy()
        if self.thumb_direction_fit:
            self.pose[-3:] = self.refine_thumb_tip(observed, pixels, size, root, scale, translation, previous_tip)
        self.last_time = timestamp
        joints, vertices = self.skin(self.pose[None], root, self.mesh)
        landmarks = scale * np.concatenate((joints[0], vertices[0, TIPS]))[ORDER]
        vertices = scale * vertices[0]
        self.preview_mesh = (vertices, translation) if translation[2] > .08 else None
        return landmarks, vertices

    def refine_thumb_tip(self, observed, pixels, size, root, scale, translation, previous):
        """Refine only the thumb IP pose from directions, with a low-curvature prior.

        The original fit's root, all joint centres, and other digits stay fixed.
        Unit 3D and projected directions allow the observed distal phalanx to
        be shorter than the template without hiding the length error in a curl.
        """
        observed_direction = unit(observed[4] - observed[3])
        image_direction = unit(pixels[4] - pixels[3])
        perturbations = np.vstack((np.zeros(3), 1e-5 * np.eye(3)))

        def residual(tip_poses):
            poses = np.broadcast_to(self.keypoint_pose, (len(tip_poses), 45)).copy()
            poses[:, -3:] = tip_poses
            joints, tips = self.skin(poses, root, self.tips)
            fitted = scale * np.concatenate((joints, tips), axis=1)[:, ORDER]
            middle = unit(fitted[:, 3] - fitted[:, 2])
            distal = unit(fitted[:, 4] - fitted[:, 3])
            terms = [(distal - observed_direction) / .35,
                     (distal - middle) / .35,
                     .2 * (tip_poses - previous)]
            if translation[2] > .08:
                camera = fitted[:, 3:5] + translation
                projected = camera[..., :2] / np.maximum(camera[..., 2:], .05)
                direction = unit(projected[:, 1] - projected[:, 0])
                terms.append((direction - image_direction) / .12)
            return np.concatenate(terms, axis=1)

        def jacobian(pose):
            values = residual(pose + perturbations)
            return (values[1:] - values[0]).T / 1e-5

        result = least_squares(lambda pose: residual(pose[None])[0], self.keypoint_pose[-3:],
                               jac=jacobian, loss='soft_l1', max_nfev=12)
        return result.x

    def draw_overlay(self, image):
        """Project the current fit using the same camera model as the 2D residual."""
        if self.preview_mesh is None:
            return False
        vertices, translation = self.preview_mesh
        camera = vertices + translation
        if not np.isfinite(camera).all() or camera[:, 2].min() <= 0:
            return False
        height, width = image.shape[:2]
        size = np.array([width, height])
        pixels = max(width, height) * camera[:, :2] / camera[:, 2:] + size / 2
        lo = np.clip(np.floor(pixels.min(axis=0)) - 1, 0, size).astype(int)
        hi = np.clip(np.ceil(pixels.max(axis=0)) + 2, 0, size).astype(int)
        if np.any(hi <= lo):
            return False
        roi = image[lo[1]:hi[1], lo[0]:hi[0]]
        layer = roi.copy()
        triangles = np.rint(pixels - lo).astype(np.int32)[self.faces]
        a, b, c = camera[self.faces].transpose(1, 0, 2)
        normals = np.cross(b - a, c - a)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
        light = np.clip(normals @ [-.30, -.43, -.85], 0, 1)
        colors = ((.55 + .45 * light[:, None]) * [225, 210, 55]).astype(np.uint8).tolist()
        # Paint near fingers last so their surface remains visible during opposition.
        for i in np.argsort(a[:, 2] + b[:, 2] + c[:, 2])[::-1]:
            cv2.fillConvexPoly(layer, triangles[i], colors[i], lineType=cv2.LINE_AA)
        cv2.addWeighted(layer, .48, roi, .52, 0, dst=roi)
        return True


class BotyardRetargeter:
    def __init__(self, solver, model, meshes, side, template, smoothing, *, gradient="finite_difference",
                 shape_model='baseline'):
        if shape_model not in ('v4_2', 'sweep', 'direction', 'baseline', 'phalanx', 'thumb_span'):
            raise ValueError(f'Unknown Botyard shape objective: {shape_model}')
        self.shape_model = shape_model
        self.opposition = 'v4_2'
        self.algorithm_version = ALGORITHM_VERSION
        self.gradient = gradient
        self.mano = OnlineMano(template or Path.home() / '.cache/botyard-retargeting/mano' / f'{side}.npz',
                               thumb_direction_fit=shape_model in ('v4_2', 'sweep'))
        self.frame_transform = (OPERATOR2MANO_RIGHT if side == 'right' else OPERATOR2MANO_LEFT) @ MANO_TO_GS_S1
        self.reference = ConstrainedRetargeter(solver, model)
        self.output = ConstrainedRetargeter(solver, model)
        self.hand = Hand(self.reference, meshes, side)
        self.closure = configure_closure(self.output, .006, True)
        neutral = self.hand.states(self.reference.last)[0][0]
        self.robot_width = np.linalg.norm(neutral[1, 0] - neutral[4, 0])
        self.palm_axes = palm_frame(np.zeros(3), *neutral[[1, 2, 4], 0])[1]
        self.filters = [LandmarkFilter(smoothing) for _ in range(4)]
        self.protected_closure = (ProtectedSurfaceClosure(ConstrainedRetargeter(solver, model), self.hand)
                                  if shape_model == 'v4_2' else None)

    def target_frame(self, joints):
        if self.shape_model in ('v4_2', 'direction', 'sweep'):
            # The unit directions must be expressed exactly as the visible MANO
            # overlay, including the neutral robot palm's anatomical axes.
            return palm_frame(*joints[[0, 5, 9, 17]])[1] @ self.palm_axes.T
        return SingleHandDetector.estimate_frame_from_hand_points(joints) @ self.frame_transform

    def solve_camera(self, camera_points, pixels, size, timestamp, dt, *, trace=None):
        joints, vertices = self.mano.fit(camera_points, pixels, size, timestamp)
        if trace is not None:
            trace.update(image_size_px=size, skeleton_px=pixels, skeleton_wrist_m=camera_points,
                         mano_joints_wrist_m=joints, mano_vertices_wrist_m=vertices,
                         mano_camera_translation_m=(self.mano.preview_mesh[1] if self.mano.preview_mesh else None))
            trace['mano_thumb_fit'] = ('directions_minimum_curvature' if self.mano.thumb_direction_fit
                                       else 'keypoint_positions')
        return self.solve_mano(joints, vertices, timestamp, dt, trace=trace)

    def solve_mano(self, joints, vertices, timestamp, dt, *, trace=None):
        """Retarget a fitted MANO frame in wrist-centered camera coordinates."""
        frame = self.target_frame(joints)
        local = (joints - joints[0]) @ frame
        if self.shape_model in ('v4_2', 'direction', 'sweep'):
            chain = self.filters[0].update(local, timestamp)[1:].reshape(5, 4, 3)
            sweep = self.shape_model in ('v4_2', 'sweep')
            previous = self.output.last.copy()
            if sweep:
                self.reference.last = self.output.last.copy()
            shape = direction_objective(self.hand, chain, self.reference.last.copy(), gradient=self.gradient,
                                        thumb_base_only=sweep)
            seed = self.hand.thumb_bend_seed(chain, self.reference.last) if sweep else None
            primary_failure = None
            try:
                q = self.reference.solve_objective(shape, dt, initial=seed)
                if sweep:
                    opponents = self.hand.states(self.reference.last)[1][0, 1:].copy()
                    shape = thumb_base_objective(self.hand, chain, self.output.last.copy(), gradient=self.gradient,
                                                 opponents=opponents)
                    names = list(self.output.opt.target_joint_names)
                    free = {names.index('THJ4'), names.index('THJ3')}
                    fixed = {i: value for i, value in enumerate(self.reference.last) if i not in free}
                    q = self.output.solve_objective(shape, dt, initial=self.reference.last, fixed=fixed)
                else:
                    self.output.last = self.reference.last.copy()
            except SOLVER_FAILURES as error:
                if self.protected_closure is None:
                    raise
                self.reference.last = previous.copy()
                self.output.last = previous.copy()
                self.output.held += 1
                primary_failure = type(error).__name__
                q = self.output.full_pose(previous)
            if trace is not None:
                trace.update(shape_model=self.shape_model, solve_dt_s=dt, target_distal_gaps_m=None,
                             reference_joint_names=self.reference.opt.target_joint_names,
                             reference_position_rad=self.reference.last.copy(),
                             reference_holds_total=self.reference.held, output_holds_total=self.output.held)
                if sweep:
                    trace['thumb_target_direction'] = self.hand.thumb_direction_target(chain)
                    trace['thumb_base_fallbacks_total'] = self.output.fallbacks
            if self.protected_closure is not None:
                primary = self.output.last.copy()
                gaps = mano_surface_gaps(dict(faces=self.mano.faces, distal_face_labels=self.mano.labels,
                                             vertices_camera_wrist_m=vertices[None]), None)[0]
                width = np.linalg.norm(joints[5] - joints[17])
                weights = np.exp(-(gaps / (.15 * width))**2)
                if primary_failure is None:
                    command, info = self.protected_closure.refine(primary, chain, weights, previous, dt)
                    self.output.last = command.copy()
                    q = self.output.full_pose(command)
                else:
                    info = dict(optimizer_exception=None, min_protection_margin=None)
                if trace is not None:
                    trace.update(algorithm='botyard', algorithm_version=self.algorithm_version,
                                 botyard_opposition=self.opposition,
                                 primary_optimizer_exception=primary_failure,
                                 closure_optimizer_exception=info['optimizer_exception'],
                                 closure_min_protection_margin=info['min_protection_margin'],
                                 closure_primary_position_rad=primary, closure_weights=weights,
                                 mano_distal_gaps_m=gaps,
                                 target_distal_gaps_m=np.full(4, self.protected_closure.distance.clearance))
            return q
        surface = dict(frame=[0], vertices=vertices[None], joints=joints[None], faces=self.mano.faces)
        pads, normals, _ = surface_targets(surface, dict(frame=[0], raw_human_palm_m=local[None]))
        gaps = mano_surface_gaps(dict(faces=self.mano.faces, distal_face_labels=self.mano.labels,
                                      vertices_camera_wrist_m=vertices[None]), None)[0]
        if trace is not None:
            trace['mano_distal_gaps_m'] = gaps
        chain, pads, normals, gaps = [f.update(v, timestamp) for f, v in zip(
            self.filters, (local, pads[0], normals[0], gaps[:, None]))]
        prev_ref = self.reference.last.copy()
        chain = chain[1:].reshape(5, 4, 3)
        normals = unit(normals)
        track_span = self.shape_model == 'thumb_span'
        shape = objective(self.hand, chain, pads, normals, prev_ref,
                          weight=0., opposition=True, gradient=self.gradient, phalanx=self.shape_model != 'baseline',
                          thumb_span=track_span)
        self.reference.solve_objective(shape, dt)
        reference = self.reference.last
        human_width = np.linalg.norm(local[5] - local[17])
        target = np.hypot(self.closure.clearance, self.robot_width / human_width * gaps.ravel())
        thumb_loss = self.hand.thumb_refinement_loss(chain, normals[0]) if track_span else None
        closure = self.closure.objective(reference, closure_weights(gaps.ravel(), .70), target,
                                         thumb_pose_loss=thumb_loss)
        q = self.output.solve_objective(closure, dt, initial=reference + (self.output.last - prev_ref))
        if trace is not None:
            trace.update(shape_model=self.shape_model, solve_dt_s=dt, target_distal_gaps_m=target,
                         reference_joint_names=self.reference.opt.target_joint_names,
                         reference_position_rad=reference, reference_holds_total=self.reference.held,
                         output_holds_total=self.output.held)
        return q


def make_botyard_retargeter(*args, opposition='v4_2', **kwargs):
    """Select the reviewed inward experiment explicitly; keep v4.2 by default."""
    if opposition == 'v4_2':
        return BotyardRetargeter(*args, **kwargs)
    if opposition != 'inward':
        raise ValueError(f'Unknown Botyard opposition mode: {opposition}')
    if kwargs.get('shape_model', 'baseline') != 'v4_2':
        raise ValueError('Inward opposition requires the v4_2 shape objective')
    # Lazy import avoids a cycle and runs the exact class/settings reviewed in
    # the offline comparison. No volar-region contact target is enabled here.
    from opposition_guidance_experiment import GuidedRetargeter
    return GuidedRetargeter(*args, **kwargs, guide_weight=.2, starts=3,
                            mode='distance_priority', facing_target='inward')
