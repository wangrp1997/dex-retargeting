"""Causal MANO fitting and the Botyard two-stage GS S1 objective."""
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from distal_closure import closure_weights, mano_surface_gaps
from fit_example_mano import ORDER, TIPS, distal_face_labels
from gs_s1_filter import LandmarkFilter
from gs_s1_optimization import ConstrainedRetargeter
from gs_s1_rviz import MANO_TO_GS_S1
from palm_anchor_experiment import Hand, objective, surface_targets, unit
from single_hand_detector import OPERATOR2MANO_LEFT, OPERATOR2MANO_RIGHT, SingleHandDetector
from sweep_closure import configure_closure


class OnlineMano:
    """Fit the pose of the local licensed MANO template to one RGB observation at a time."""

    def __init__(self, path):
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
        self.tips = (self.template[TIPS], self.weights[TIPS], self.directions[TIPS])
        self.mesh = (self.template, self.weights, self.directions)
        self.perturbations = np.vstack((np.zeros(45), 1e-5 * np.eye(45)))
        self.scale_filter = LandmarkFilter('one_euro')
        self.last_time = None

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
        previous = self.mean if fresh else self.pose

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
        self.pose = result.x
        self.last_time = timestamp
        joints, vertices = self.skin(self.pose[None], root, self.mesh)
        return scale * np.concatenate((joints[0], vertices[0, TIPS]))[ORDER], scale * vertices[0]


class BotyardRetargeter:
    def __init__(self, solver, model, meshes, side, template, smoothing):
        self.mano = OnlineMano(template or Path.home() / '.cache/botyard-retargeting/mano' / f'{side}.npz')
        self.frame_transform = (OPERATOR2MANO_RIGHT if side == 'right' else OPERATOR2MANO_LEFT) @ MANO_TO_GS_S1
        self.reference = ConstrainedRetargeter(solver, model)
        self.output = ConstrainedRetargeter(solver, model)
        self.hand = Hand(self.reference, meshes, side)
        self.closure = configure_closure(self.output, .006, True)
        neutral = self.hand.states(self.reference.last)[0][0]
        self.robot_width = np.linalg.norm(neutral[1, 0] - neutral[4, 0])
        self.filters = [LandmarkFilter(smoothing) for _ in range(4)]

    def solve_camera(self, camera_points, pixels, size, timestamp, dt):
        joints, vertices = self.mano.fit(camera_points, pixels, size, timestamp)
        frame = SingleHandDetector.estimate_frame_from_hand_points(joints) @ self.frame_transform
        local = (joints - joints[0]) @ frame
        surface = dict(frame=[0], vertices=vertices[None], joints=joints[None], faces=self.mano.faces)
        pads, normals, _ = surface_targets(surface, dict(frame=[0], raw_human_palm_m=local[None]))
        gaps = mano_surface_gaps(dict(faces=self.mano.faces, distal_face_labels=self.mano.labels,
                                      vertices_camera_wrist_m=vertices[None]), None)[0]
        chain, pads, normals, gaps = [f.update(v, timestamp) for f, v in zip(
            self.filters, (local, pads[0], normals[0], gaps[:, None]))]
        prev_ref = self.reference.last.copy()
        shape = objective(self.hand, chain[1:].reshape(5, 4, 3), pads, unit(normals), prev_ref,
                          weight=0., opposition=True)
        self.reference.solve_objective(shape, dt)
        reference = self.reference.last
        human_width = np.linalg.norm(local[5] - local[17])
        target = np.hypot(self.closure.clearance, self.robot_width / human_width * gaps.ravel())
        closure = self.closure.objective(reference, closure_weights(gaps.ravel(), .70), target)
        return self.output.solve_objective(closure, dt, initial=reference + (self.output.last - prev_ref))
