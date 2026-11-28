"""Botyard v4.3 inward opposition and earlier offline comparison objectives.

Replays cached MANO fits via validate_botyard. No camera, ROS node, motion
publisher or hardware connection is created by this CLI. This borrows directional contact
and multiple-start ideas, not ContactOpt's network or tissue penetration model.
"""
import argparse
from functools import partial

import numpy as np
from pinocchio import hppfcl

from botyard_live import BotyardRetargeter
from distal_closure import DistalClosure
from gs_s1_optimization import ConstrainedRetargeter
from gs_s1_botyard_version import botyard_version
from palm_anchor_experiment import ANGLE, LENGTH
from shape_priority_closure import ProtectedSurfaceClosure, SOLVER_FAILURES, distal_gap_cost


def volar_mesh(mesh, center, normal):
    """Restrict contact to an outer patch around the existing CAD pad sample.

    These fixed mesh-space bounds select geometry, not a gesture or a joint
    angle. They cover +/-6 mm laterally and +/-7.5 mm along the distal segment,
    exclude facets more than 2 mm behind the pad, and require outward normals
    within arccos(0.4) of its sampled normal. Full CAD remains in collisions.
    """
    vertices = mesh.vertices()
    indices = np.array([[mesh.tri_indices(i)[j] for j in range(3)]
                        for i in range(mesh.num_tris)])
    triangles = vertices[indices]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-15)
    shift = triangles.mean(axis=1) - center
    selected = ((normals @ normal > .4) & (shift @ normal > -.002)
                & (abs(shift[:, 0]) < .006) & (abs(shift[:, 2]) < .0075))
    count = int(selected.sum())
    if count < 10:
        raise ValueError(f'CAD volar patch has too few triangles: {count}')
    patch = hppfcl.BVHModelOBBRSS()
    patch.beginModel(count, 3 * count)
    for triangle in triangles[selected]:
        patch.addTriangle(*triangle)
    patch.endModel()
    return patch, count


def facing_residual(closure, q):
    """Continuous robot pad-direction residuals and actuator derivatives."""
    _, pads, normals, jp, jn, _, _ = closure.kinematics(q)
    delta = pads[1:] - pads[0]
    # Numerical regularization at coincident centres, not contact classification.
    length = np.sqrt(np.sum(delta**2, axis=1) + 1e-12)
    direction = delta / length[:, None]
    tangent = np.eye(3)[None] - direction[:, :, None] * direction[:, None, :]
    jac = np.einsum('fcd,fdn->fcn', tangent, jp[1:] - jp[0]) / length[:, None, None]
    return normals[0] - direction, jn[0] - jac


class GuidedClosure(ProtectedSurfaceClosure):
    """Earlier, rejected volar-region experiment, retained for reproducibility."""

    def __init__(self, solver, hand, *, guide_weight=.1, starts=3):
        if not np.isfinite(guide_weight) or guide_weight < 0 or starts not in (1, 2, 3):
            raise ValueError('Expected nonnegative finite guidance and 1, 2 or 3 starts')
        super().__init__(solver, hand)
        self.guide_weight, self.starts = guide_weight, starts
        self.pad_distance = DistalClosure(solver)
        self.pad_distance.meshes[0], self.pad_facets = volar_mesh(
            self.pad_distance.meshes[0], hand.centers[0], hand.normals[0])
        self.base = np.array([self.names.index(name) for name in ('THJ4', 'THJ3')])

    def objective(self, q, weights):
        value, slope = distal_gap_cost(self.pad_distance, q, self.qref, weights)
        residual, jac = facing_residual(self, q)
        value += self.guide_weight * float(weights @ np.sum(residual**2, axis=1)) / ANGLE**2
        slope += 2 * self.guide_weight * np.einsum('f,fc,fcn->n', weights, residual, jac) / ANGLE**2
        field, jfield = self.field_error(q)
        importance = .02 * float(np.sum(weights)) / ANGLE**2
        return value + importance * field, slope + importance * jfield

    def initial_candidates(self, primary, best, weights):
        seeds = [best.copy()]
        # The CAD neutral base is an inward-facing search candidate, not a
        # commanded posture. Clipping and every nonlinear protection still apply.
        neutral = primary.copy()
        neutral[self.base] = 0.
        seeds.append(np.clip(neutral, self.lower, self.upper))
        if hasattr(self, 'direction_targets'):
            _, _, normals, _, jn, _, _ = self.kinematics(primary)
            slope = ((weights @ (normals[0] - self.direction_targets)) @ jn[0])[self.base]
        else:
            residual, jac = facing_residual(self, primary)
            slope = np.einsum('f,fc,fcn->n', weights, residual, jac)[self.base]
        guide = primary.copy()
        scale = max(float(np.max(abs(slope))), 1e-12)
        guide[self.base] -= .5 * (self.upper - self.lower)[self.base] * slope / scale
        seeds.append(np.clip(guide, self.lower, self.upper))
        unique = []
        for seed in seeds[:self.starts]:
            if not any(np.array_equal(seed, prior) for prior in unique):
                unique.append(seed)
        return unique

    def refine(self, primary, chain, weights, previous, dt):
        self.setup(primary, chain, previous, dt)
        if not self.feasible(primary):
            raise ValueError('Primary pose must be feasible from the previous command')
        best = primary.copy()
        best_value = self.objective(best, weights)[0]

        def objective(q, grad):
            nonlocal best, best_value
            value, slope = self.objective(q, weights)
            if grad.size:
                grad[:] = slope
            if value < best_value and self.feasible(q):
                best, best_value = q.copy(), value
            return value

        objective(previous, np.empty(0))
        failures = []
        limits = self.solver.limits
        self.solver.limits = self.frame_limits
        try:
            for seed in self.initial_candidates(primary, best, weights):
                # Every start gets the same physical previous command and dt.
                # Restarts cannot accumulate a larger velocity allowance.
                self.solver.last = previous.copy()
                try:
                    self.solver.solve_objective(objective, dt, initial=seed)
                except SOLVER_FAILURES as error:
                    failures.append(type(error).__name__)
                objective(self.solver.last, np.empty(0))
        finally:
            self.solver.limits = limits
            self.solver.last = best.copy()
        return best, dict(optimizer_exception=','.join(failures) if failures else None,
                          min_protection_margin=float(self.protection(best)[0].min()))


class DistancePriorityClosure(ProtectedSurfaceClosure):
    """Preserve full-link closure first, then guide the thumb's pad normal.

    The first solve is the unchanged v4.2 objective. The second solve has four
    separate proximity-weighted upper bounds on those full-STL distances. Its
    direction targets are frozen at the first result: moving the other finger
    cannot itself reduce the directional objective. All starts retain the
    original frame's previous command, velocity bounds and shape protections.
    """

    def __init__(self, solver, hand, *, guide_weight=.1, starts=3, gap_slack_m=.0001,
                 facing_target='inward'):
        super().__init__(solver, hand)
        if not np.isfinite(guide_weight) or guide_weight < 0 or starts not in (1, 2, 3):
            raise ValueError('Expected nonnegative finite guidance and 1, 2 or 3 starts')
        if not np.isfinite(gap_slack_m) or gap_slack_m < 0:
            raise ValueError('Expected a finite nonnegative gap allowance')
        if facing_target not in ('inward', 'opponent'):
            raise ValueError('Unknown thumb-facing reference')
        self.guide_weight, self.starts = guide_weight, starts
        self.facing_target = facing_target
        self.gap_slack_m = gap_slack_m
        self.guiding, self.gap_reference = False, None
        self.base = np.array([self.names.index(name) for name in ('THJ4', 'THJ3')])
        self.solver.minimizer.add_inequality_mconstraint(self.distance_inequalities, [1e-7] * 4)
        # This patch is measured for comparison only, never used by either solve.
        self.pad_distance = DistalClosure(solver)
        self.pad_distance.meshes[0], self.pad_facets = volar_mesh(
            self.pad_distance.meshes[0], hand.centers[0], hand.normals[0])

    def distance_inequalities(self, result, q, grad):
        if self.gap_reference is None:
            result[:] = -1.
            if grad.size:
                grad[:] = 0.
            return
        gaps, jac = self.distance.distances(q, with_gradient=bool(grad.size))
        result[:] = (self.gap_scale * (gaps - self.gap_reference) - self.gap_slack_m) / LENGTH
        if grad.size:
            grad[:] = self.gap_scale[:, None] * jac / LENGTH

    def feasible(self, q):
        if not super().feasible(q):
            return False
        if self.gap_reference is None:
            return True
        margins = np.empty(4)
        self.distance_inequalities(margins, q, np.empty(0))
        return bool(margins.max() <= 1e-7)

    def objective(self, q, weights):
        if not self.guiding:
            return super().objective(q, weights)
        _, _, normals, _, jn, _, _ = self.kinematics(q)
        error = normals[0] - self.direction_targets
        value = self.guide_weight * float(weights @ np.sum(error**2, axis=1)) / ANGLE**2
        slope = 2 * self.guide_weight * (weights @ error) @ jn[0] / ANGLE**2
        shift = q - self.closure_reference
        # Keep the anchor independent of proximity confidence. Scaling both
        # terms by confidence would let tiny nonzero weights select a visibly
        # different search seed even on an open hand.
        anchor = .03 / ANGLE**2
        return value + anchor * float(shift @ shift), slope + 2 * anchor * shift

    def refine(self, primary, chain, weights, previous, dt):
        self.guiding, self.gap_reference = False, None
        baseline, info = super().refine(primary, chain, weights, previous, dt)
        self.closure_reference = baseline.copy()
        self.gap_reference = self.distance.distances(baseline)[0].copy()
        # Smooth normalization, with a numerical zero-weight regularizer.
        self.gap_scale = weights / (float(weights.sum()) + 1e-12)
        if self.facing_target == 'inward':
            # Reference only: the CAD's initial base orientation with THIS
            # frame's flexion retained. No pad-centre alignment is requested,
            # and none of these reference joint angles are commanded/fixed.
            neutral_base = baseline.copy()
            neutral_base[self.base] = 0.
            reference = self.kinematics(neutral_base)[2][0]
            reference = reference / np.linalg.norm(reference)
            self.direction_targets = np.broadcast_to(reference, (4, 3)).copy()
        else:
            _, pads, _, _, _, _, _ = self.kinematics(baseline)
            delta = pads[1:] - pads[0]
            self.direction_targets = delta / np.sqrt(np.sum(delta**2, axis=1) + 1e-12)[:, None]
        self.guiding = True
        best, best_value = baseline.copy(), self.objective(baseline, weights)[0]
        # All terms are nonnegative: no alternative can improve this result
        # by more than the solver's existing objective stopping precision.
        if best_value <= self.solver.minimizer.get_ftol_abs():
            return baseline, info

        def objective(q, grad):
            nonlocal best, best_value
            value, slope = self.objective(q, weights)
            if grad.size:
                grad[:] = slope
            if value < best_value and self.feasible(q):
                best, best_value = q.copy(), value
            return value

        seeds = GuidedClosure.initial_candidates(self, baseline, baseline, weights)
        failures = [info['optimizer_exception']] if info['optimizer_exception'] else []
        limits = self.solver.limits
        self.solver.limits = self.frame_limits
        try:
            for seed in seeds:
                self.solver.last = previous.copy()
                try:
                    self.solver.solve_objective(objective, dt, initial=seed)
                except SOLVER_FAILURES as error:
                    failures.append(type(error).__name__)
                objective(self.solver.last, np.empty(0))
        finally:
            self.solver.limits = limits
            self.solver.last = best.copy()
        return best, dict(optimizer_exception=','.join(failures) if failures else None,
                          min_protection_margin=float(self.protection(best)[0].min()))


class GuidedRetargeter(BotyardRetargeter):
    def __init__(self, solver, model, meshes, side, template, smoothing, *, gradient='analytic',
                 shape_model='v4_2', guide_weight=.1, starts=3, mode='distance_priority',
                 facing_target='inward'):
        if shape_model != 'v4_2':
            raise ValueError('The experiment requires the v4.2 primary controller')
        super().__init__(solver, model, meshes, side, template, smoothing,
                         gradient=gradient, shape_model=shape_model)
        if mode not in ('distance_priority', 'volar'):
            raise ValueError('Unknown offline guidance experiment')
        closure = DistancePriorityClosure if mode == 'distance_priority' else GuidedClosure
        settings = dict(guide_weight=guide_weight, starts=starts)
        if mode == 'distance_priority':
            settings['facing_target'] = facing_target
        self.protected_closure = closure(ConstrainedRetargeter(solver, model), self.hand, **settings)
        self.opposition = ('inward' if mode == 'distance_priority' and facing_target == 'inward'
                           else mode + '_' + facing_target)
        self.algorithm_version = botyard_version(self.opposition) if self.opposition == 'inward' else 'experimental'


def contact_metrics(ret, candidates):
    result = {}
    for label, q in candidates.items():
        _, pads, normals = (state[0] for state in ret.hand.states(q))
        delta = pads[1:] - pads[0]
        length = np.maximum(np.linalg.norm(delta, axis=1), 1e-12)
        facing = np.degrees(np.arccos(np.clip(delta @ normals[0] / length, -1., 1.)))
        opposing = np.degrees(np.arccos(np.clip(-normals[1:] @ normals[0], -1., 1.)))
        gaps = ret.protected_closure.pad_distance.distances(q)[0]
        for pair in range(4):
            result[f'{label}_facing_{pair}_deg'] = float(facing[pair])
            result[f'{label}_opposing_normal_{pair}_deg'] = float(opposing[pair])
            result[f'{label}_pad_gap_{pair}_mm'] = float(gaps[pair] * 1000)
        closure = ret.protected_closure
        if isinstance(closure, DistancePriorityClosure):
            reference = getattr(closure, 'direction_targets', normals[0][None])[0]
            result[f'{label}_inward_reference_error_deg'] = float(np.degrees(
                np.arccos(np.clip(normals[0] @ reference, -1., 1.))))
    closure = ret.protected_closure
    if isinstance(closure, DistancePriorityClosure):
        for pair in range(4):
            result[f'same_frame_v4_2_gap_{pair}_mm'] = (float(closure.gap_reference[pair] * 1000)
                                                       if closure.gap_reference is not None else float('nan'))
        margins = np.empty(4)
        closure.distance_inequalities(margins, candidates['validation'], np.empty(0))
        result['max_distance_priority_violation'] = float(margins.max())
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--guide-weight', type=float, default=.1)
    parser.add_argument('--starts', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--mode', choices=('distance_priority', 'volar'), default='distance_priority')
    parser.add_argument('--facing-target', choices=('inward', 'opponent'), default='inward')
    settings, remaining = parser.parse_known_args()
    import sys
    # Forward only the validator's arguments, retaining its help/validation.
    sys.argv = [sys.argv[0], *remaining]
    from validate_botyard import main as validate
    validate(retargeter_factory=partial(GuidedRetargeter, guide_weight=settings.guide_weight,
                                        starts=settings.starts, mode=settings.mode,
                                        facing_target=settings.facing_target), extra_metrics=contact_metrics,
             metadata=dict(experiment=settings.mode + '_multistart', live_default_changed=False,
                           guide_weight=settings.guide_weight, starts=settings.starts,
                           protections='v4.2 source and physical bounds retained',
                           gap_slack_m=.0001 if settings.mode == 'distance_priority' else None,
                           facing_target=settings.facing_target if settings.mode == 'distance_priority' else 'opponent',
                           fixed_direction_targets=settings.mode == 'distance_priority'))


if __name__ == '__main__':
    main()
