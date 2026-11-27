"""Botyard v4.2 source-referenced shape priority and distal surface closure.

The primary solve still fits MANO directions/bends and the custom thumb-facing
field. Bind these constraints to its feasible result on every frame. A secondary
solve may then reduce distal surface gaps without spending the same local pose
allowance repeatedly and eventually erasing a fist.
"""

import mujoco
import nlopt
import numpy as np

from distal_closure import DistalClosure
from palm_anchor_experiment import ANGLE, JOINT_MOTION_WEIGHT, LENGTH, bone_geometry

ALGORITHM_VERSION = '4.2'
SOLVER_FAILURES = (nlopt.RoundoffLimited, RuntimeError, nlopt.runtime_error,
                   nlopt.invalid_argument)


class SourceShapePriority:
    """Separate source-error bounds for five bends and four proximal directions.

    Angular tolerances are absolute errors against the current source, rather
    than increments around the previous command. Above those tolerances the
    secondary task cannot undo progress made by the primary solve. Each digit
    has its own bound; errors cannot be traded between fingers. The thumb's
    absolute direction is deliberately absent because it has a custom policy.
    """

    def __init__(self, hand, *, bend_tolerance_deg=10., root_tolerance_deg=10.):
        if bend_tolerance_deg < 0 or root_tolerance_deg < 0:
            raise ValueError('Shape tolerances must be nonnegative')
        self.hand = hand
        self.n = hand.mapping.shape[1]
        self.floor = np.r_[
            np.full(5, (np.deg2rad(bend_tolerance_deg) / ANGLE)**2),
            np.full(4, 2 * (1 - np.cos(np.deg2rad(root_tolerance_deg))) / ANGLE**2)]
        self.jp = np.empty((3, hand.model.nv))
        self.jr = np.empty_like(self.jp)
        self.cached_q = None

    def bind(self, chain, primary):
        self.target_bones, _, _, _, self.target_bend = bone_geometry(chain)
        self.cached_q = None
        self.limits = np.maximum(self.losses(primary)[0], self.floor)
        return self

    def losses(self, q):
        if self.cached_q is not None and np.array_equal(q, self.cached_q):
            return self.cached_losses
        hand = self.hand
        pos = hand.states(q)[0][0]
        mujoco.mj_comPos(hand.model, hand.data)
        jpos = np.empty((5, 4, 3, self.n))
        for digit in range(5):
            for bone in range(4):
                mujoco.mj_jacBody(hand.model, hand.data, self.jp, self.jr,
                                  hand.ids[digit, bone])
                jpos[digit, bone] = hand.axes.T @ self.jp @ hand.mapping
        bones, lengths, axis, sine, bend = bone_geometry(pos)
        delta_jac = np.diff(jpos, axis=1)
        jbone = (delta_jac - bones[..., None] * np.einsum(
            'dbc,dbcn->dbn', bones, delta_jac)[..., None, :]) / lengths[..., None]
        plane = np.divide(axis, sine[..., None], out=np.zeros_like(axis),
                          where=sine[..., None] > 64 * np.finfo(float).eps)
        jbend = (np.einsum('dbc,dbcn->dbn', np.cross(bones[:, :-1], plane), jbone[:, :-1])
                 + np.einsum('dbc,dbcn->dbn', np.cross(plane, bones[:, 1:]), jbone[:, 1:]))
        bend_error = bend - self.target_bend
        root_error = bones[1:, 0] - self.target_bones[1:, 0]
        values = np.r_[np.mean(bend_error**2, axis=1),
                       np.sum(root_error**2, axis=1)] / ANGLE**2
        slopes = np.vstack((np.einsum('db,dbn->dn', bend_error, jbend),
                            2 * np.einsum('dc,dcn->dn', root_error, jbone[1:, 0]))) / ANGLE**2
        self.cached_q = q.copy()
        self.cached_losses = values, slopes
        return self.cached_losses

    def margins(self, q):
        """Return nonnegative feasible margins and their actuator Jacobians."""
        values, slopes = self.losses(q)
        return self.limits + 1e-9 - values, -slopes


def distal_gap_cost(distance, q, primary, weights, *, anchor=JOINT_MOTION_WEIGHT):
    """Reduce any distal-surface separation; stop at the collision clearance.

    ``weights`` are continuous MANO proximity weights supplied by the caller.
    The squared positive-part residual is differentiable at clearance. Mesh
    nearest-point changes remain piecewise differentiable. No pad centre or
    pad-normal matching is imposed, and human distances are not robot targets.
    """
    gaps, jac = distance.distances(q, with_gradient=True)
    residual = np.maximum(gaps - distance.clearance, 0.) / LENGTH
    value = float(weights @ residual**2)
    slope = (2 * weights * residual / LENGTH) @ jac
    shift = q - primary
    value += anchor * float(np.sum((shift / ANGLE)**2))
    slope += 2 * anchor * shift / ANGLE**2
    return value, slope


class ProtectedSurfaceClosure:
    """Validated v4.2 secondary task, with a separate constrained solver.

    The caller supplies the feasible primary result, current filtered MANO
    chain, continuous proximity weights, previous actual command and frame dt.
    Feasible evaluated candidates survive numerical solver failures. The
    primary pose remains the fallback, subject to the same physical bounds.
    """

    def __init__(self, solver, hand):
        self.solver, self.hand = solver, hand
        self.geom, self.palm = solver.geom, solver.palm
        self.distance = DistalClosure(solver)
        self.source_priority = SourceShapePriority(hand)
        self.names = list(solver.opt.target_joint_names)
        self.n = len(self.names)
        fingers = ('FF', 'MF', 'RF', 'LF')
        self.spread = np.array([self.names.index(f + 'J4') for f in fingers])
        self.flex = np.array([self.names.index(n) for n in ('THJ2', 'THJ1')]
                             + [self.names.index(f + j) for f in fingers for j in ('J3', 'J2')])
        self.jp = np.empty((3, hand.model.nv))
        self.jr = np.empty_like(self.jp)
        self.cached_q = None
        solver.minimizer.add_inequality_mconstraint(self.inequalities, [1e-7] * 14)
        solver.minimizer.set_maxeval(80)

    def kinematics(self, q):
        if self.cached_q is not None and np.array_equal(q, self.cached_q):
            return self.cached_state
        hand = self.hand
        pos, pads, normals = (a[0] for a in hand.states(q))
        mujoco.mj_comPos(hand.model, hand.data)
        jp = np.empty((5, 3, self.n))
        jn = np.empty_like(jp)
        for digit in range(5):
            point = hand.wrist + hand.axes @ pads[digit]
            mujoco.mj_jac(hand.model, hand.data, self.jp, self.jr, point, hand.ids[digit, 2])
            jp[digit] = hand.axes.T @ self.jp @ hand.mapping
            angular = hand.axes.T @ self.jr @ hand.mapping
            jn[digit] = np.cross(angular.T, normals[digit]).T
        delta = pos[0, -1] - pos[0, 0]
        length = np.linalg.norm(delta)
        ray = delta / length
        mujoco.mj_jacBody(hand.model, hand.data, self.jp, self.jr, hand.ids[0, -1])
        jtip = hand.axes.T @ self.jp @ hand.mapping
        mujoco.mj_jacBody(hand.model, hand.data, self.jp, self.jr, hand.ids[0, 0])
        jroot = hand.axes.T @ self.jp @ hand.mapping
        jray = (np.eye(3) - np.outer(ray, ray)) @ (jtip - jroot) / length
        self.cached_q = q.copy()
        self.cached_state = pos, pads, normals, jp, jn, ray, jray
        return self.cached_state

    def field_error(self, q):
        _, pads, normals, jp, jn, _, _ = self.kinematics(q)
        delta = self.opponents - pads[0]
        square = np.sum(delta**2, axis=-1) + LENGTH**2
        field = np.sum(delta / square[:, None], axis=0)
        length = max(np.linalg.norm(field), 1e-12)
        direction = field / length
        slope = np.sum(-np.eye(3)[None] / square[:, None, None]
                       + 2 * delta[:, :, None] * delta[:, None, :] / square[:, None, None]**2, axis=0)
        jac = (np.eye(3) - np.outer(direction, direction)) @ slope @ jp[0] / length
        error = normals[0] - direction
        return float(error @ error), 2 * error @ (jn[0] - jac)

    def plane_error(self, q):
        spread = q[self.spread]
        parallel, cosine, sine = self.hand.flexion_basis.transpose(1, 0, 2)
        axis = parallel + cosine * np.cos(spread[:, None]) + sine * np.sin(spread[:, None])
        derivative = -cosine * np.sin(spread[:, None]) + sine * np.cos(spread[:, None])
        residual = np.einsum('fc,fbc->fb', axis, self.source_bones)
        slopes = 2 * np.mean(residual * np.einsum('fc,fbc->fb', derivative, self.source_bones), axis=1)
        jac = np.zeros((4, self.n))
        jac[np.arange(4), self.spread] = slopes
        return np.mean(residual**2, axis=1), jac

    def setup(self, primary, chain, previous, dt):
        self.qref = primary.copy()
        self.source_bones = bone_geometry(chain[1:])[0]
        state = self.kinematics(primary)
        self.opponents, self.ref_ray = state[1][1:].copy(), state[5].copy()
        self.plane_reference = self.plane_error(primary)[0]
        self.geom.update(primary)
        self.palm.linearize(self.geom.data, self.geom.mapping)
        self.source_priority.bind(chain, primary)
        lower, upper = self.solver.limits.T.copy()
        radius = np.deg2rad(10.)
        lower[self.flex] = np.maximum(lower[self.flex], primary[self.flex] - radius)
        upper[self.flex] = np.minimum(upper[self.flex], primary[self.flex] + radius)
        self.frame_limits = np.column_stack((lower, upper))
        step = self.solver.speed * dt
        self.lower = np.maximum(lower, previous - step)
        self.upper = np.minimum(upper, previous + step)

    def protection(self, q):
        _, _, _, _, _, ray, jray = self.kinematics(q)
        cap = 2 * (1 - np.cos(np.deg2rad(15.)))
        error = ray - self.ref_ray
        plane, jplane = self.plane_error(q)
        tolerance = np.sin(np.deg2rad(1.))**2
        source, jsource = self.source_priority.margins(q)
        values = np.r_[(cap - error @ error) / cap,
                       (self.plane_reference + tolerance - plane) / tolerance, source]
        slopes = np.vstack((-2 * error @ jray / cap, -jplane / tolerance, jsource))
        return values, slopes

    def objective(self, q, weights):
        value, slope = distal_gap_cost(self.distance, q, self.qref, weights)
        # Keep the custom inward-facing field as a weak preference. A hard
        # field-error cap prevented index opposition while the three extended
        # fingers still contributed to the primary orientation field. Source
        # shape, thumb heading and all physical bounds remain hard constraints.
        field, jfield = self.field_error(q)
        importance = .02 * float(np.sum(weights)) / ANGLE**2
        return value + importance * field, slope + importance * jfield

    def inequalities(self, result, q, grad):
        values, slopes = self.protection(q)
        result[:] = -values
        if grad.size:
            grad[:] = -slopes

    def feasible(self, q):
        if not np.isfinite(q).all():
            return False
        if (q < self.lower - 1e-8).any() or (q > self.upper + 1e-8).any():
            return False
        if self.protection(q)[0].min() < -1e-5:
            return False
        return (self.geom.feasible(q)
                and self.palm.clear(self.geom.data, self.distance.clearance - 1e-7))

    def refine(self, primary, chain, weights, previous, dt):
        self.setup(primary, chain, previous, dt)
        if not self.feasible(primary):
            raise ValueError('Primary pose must be feasible from the previous command')
        best = primary.copy()
        best_value = self.objective(primary, weights)[0]

        def objective(q, grad):
            nonlocal best, best_value
            value, slope = self.objective(q, weights)
            if grad.size:
                grad[:] = slope
            if value < best_value and self.feasible(q):
                best, best_value = q.copy(), value
            return value

        limits = self.solver.limits
        self.solver.limits = self.frame_limits
        self.solver.last = previous.copy()
        failure = None
        # Preserve an established contact when the previous actual command is
        # still inside THIS frame's source-error and physical bounds. A primary
        # solve can otherwise retract a good contact before every refinement.
        # If primary tracking made progress that the old command would undo,
        # feasible() rejects it and the new primary pose remains the seed.
        objective(previous, np.empty(0))
        initial = best.copy()
        try:
            self.solver.solve_objective(objective, dt, initial=initial)
        except SOLVER_FAILURES as error:
            failure = type(error).__name__
        finally:
            self.solver.limits = limits
        objective(self.solver.last, np.empty(0))
        self.solver.last = best.copy()
        return best, dict(optimizer_exception=failure,
                          min_protection_margin=float(self.protection(best)[0].min()))
