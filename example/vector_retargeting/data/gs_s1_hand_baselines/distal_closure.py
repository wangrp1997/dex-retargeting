"""Continuous closure assistance between the complete distal surfaces of a hand."""
import numpy as np
import mujoco
from pinocchio import hppfcl

from fit_example_mano import distal_face_labels
from gs_s1_dexmv import DIGITS
from gs_s1_palm import load_body_mesh
from palm_anchor_experiment import LENGTH


def mano_surface_gaps(mano, model_path):
    """Unsigned distances in metres. Intersecting fitted surfaces have gap zero."""
    faces = mano['faces']
    if 'distal_face_labels' in mano:
        labels = mano['distal_face_labels']
    elif model_path is not None:
        with np.load(model_path) as template:
            np.testing.assert_array_equal(faces, template['f'])
            labels = (template['distal_face_labels'] if 'distal_face_labels' in template
                      else distal_face_labels(faces, template['weights']))
    else:
        raise ValueError('This MANO fit predates distal_face_labels. Pass the prepared MANO template with --mano-model.')
    if labels.shape != (len(faces),):
        raise ValueError('Incompatible MANO face labels')
    regions = [faces[labels == digit] for digit in range(5)]
    if any(len(region) == 0 for region in regions):
        raise ValueError('Expected five nonempty MANO distal regions')
    identity = hppfcl.Transform3f()
    request = hppfcl.DistanceRequest()
    distances = []
    for vertices in mano['vertices_camera_wrist_m']:
        meshes = []
        for region in regions:
            mesh = hppfcl.BVHModelOBBRSS()
            mesh.beginModel(len(region), 3 * len(region))
            for a, b, c in vertices[region]:
                mesh.addTriangle(a, b, c)
            mesh.endModel()
            meshes.append(mesh)
        distances.append([max(0., hppfcl.distance(meshes[0], identity, other, identity,
                                                  request, hppfcl.DistanceResult()))
                          for other in meshes[1:]])
    return np.asarray(distances)


def closure_weights(gaps, strength):
    # LENGTH is the same 10 mm distance scale as the shape fit. No state switch.
    return strength / np.square(1. + np.square(gaps / LENGTH))


class DistalClosure:
    def __init__(self, solver):
        self.model = solver.geom.model
        self.data = mujoco.MjData(self.model)
        self.mapping = solver.geom.mapping
        self.clearance = solver.geom.clearance
        self.ids = [self.model.body(d + 'distal').id for d in DIGITS]
        loader = hppfcl.MeshLoader()
        self.meshes = [load_body_mesh(self.model, loader, i) for i in self.ids]
        self.request = hppfcl.DistanceRequest()
        self.request.enable_nearest_points = True
        self.jac_a = np.empty((3, self.model.nv))
        self.jac_b = np.empty_like(self.jac_a)
        # Penalize changes to every physical joint, including coupled joints.
        # Twice the thumb IP angular residual gives it four times the squared cost.
        self.joint_penalties = np.sum(self.mapping**2, axis=0) / np.deg2rad(12.)**2
        thumb_ip = list(solver.opt.target_joint_names).index('THJ1')
        self.joint_penalties[thumb_ip] *= 4.

    def distances(self, x, with_gradient=False):
        self.data.qpos[:] = self.mapping @ x
        mujoco.mj_kinematics(self.model, self.data)
        if with_gradient:
            mujoco.mj_comPos(self.model, self.data)
        transforms = [hppfcl.Transform3f(self.data.xmat[i].reshape(3, 3), self.data.xpos[i])
                      for i in self.ids]
        gaps = np.empty(4)
        jac = np.zeros((4, len(x))) if with_gradient else None
        for digit in range(1, 5):
            result = hppfcl.DistanceResult()
            distance = hppfcl.distance(self.meshes[0], transforms[0], self.meshes[digit],
                                       transforms[digit], self.request, result)
            gaps[digit - 1] = max(distance, 0.)
            if with_gradient and distance > 1e-10:
                a, b = result.getNearestPoint1(), result.getNearestPoint2()
                mujoco.mj_jac(self.model, self.data, self.jac_a, None, a, self.ids[0])
                mujoco.mj_jac(self.model, self.data, self.jac_b, None, b, self.ids[digit])
                jac[digit - 1] = ((b-a) / distance) @ (self.jac_b-self.jac_a) @ self.mapping
        return gaps, jac

    def objective(self, reference, weights, target=None, *, motion=None):
        if target is None:
            target = self.clearance
        terms = [(target, weights)]
        if motion is not None:
            terms.append(motion)
        def callback(x, grad):
            delta = x - reference
            cost = np.dot(self.joint_penalties, delta**2)
            gaps, jac = self.distances(x, with_gradient=bool(grad.size))
            if grad.size:
                grad[:] = 2. * self.joint_penalties * delta
            for desired, importance in terms:
                residual = (gaps - desired) / LENGTH
                scale = np.hypot(1., residual)
                # Both residuals use the same bounded-slope pseudo-Huber loss.
                cost += np.dot(importance, 2. * (scale - 1.))
                if grad.size:
                    grad += (2. * importance * residual / (scale * LENGTH)) @ jac
            return float(cost)
        return callback
