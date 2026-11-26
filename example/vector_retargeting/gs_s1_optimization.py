"""Cartesian hand retargeting with MuJoCo mesh-separation constraints in SLSQP."""

import numpy as np
import mujoco
import nlopt
from gs_s1_palm import PalmDistances


class MeshConstraints:
    # Distances are in metres. The positive clearance leaves room for numerical
    # error. The activation band only controls which distances MuJoCo returns.
    clearance = .0005
    activation = .006

    def __init__(self, model, drive_names):
        self.model = model
        self.data = mujoco.MjData(model)
        model.geom_margin[:] = self.activation
        self.mapping = np.zeros((model.nq, len(drive_names)))
        for i, name in enumerate(drive_names):
            self.mapping[model.joint(name).qposadr.item(), i] = 1
        for child, parent in zip(model.eq_obj1id, model.eq_obj2id):
            self.mapping[model.jnt_qposadr[child]] = self.mapping[model.jnt_qposadr[parent]]
        # One inequality for the nearest pair in each finger/finger or
        # finger/palm group, including nonadjacent links of the same digit.
        digits = ["th", "ff", "mf", "rf", "lf"]
        group = [model.body(body).name[:2] for body in model.geom_bodyid]
        group = [digits.index(g) if g in digits else len(digits) for g in group]
        keys = sorted({tuple(sorted((a, b))) for a in group for b in group})
        self.rows = np.array([[keys.index(tuple(sorted((a, b)))) for b in group] for a in group])
        self.count = len(keys)
        self.jac_a = np.empty((3, model.nv))
        self.jac_b = np.empty_like(self.jac_a)

    def update(self, x):
        self.data.qpos[:] = self.mapping @ x
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

    def feasible(self, x):
        self.update(x)
        return np.min(self.data.contact.dist, initial=self.activation) >= self.clearance - 1e-7

    def inequalities(self, result, x, grad):
        self.update(x)
        result[:] = self.clearance - self.activation
        if grad.size:
            grad[:] = 0
        for contact in self.data.contact:
            i, j = contact.geom
            row = self.rows[i, j]
            value = self.clearance - contact.dist
            if value <= result[row]:
                continue
            result[row] = value
            if grad.size:
                mujoco.mj_jac(self.model, self.data, self.jac_a, None, contact.pos, self.model.geom_bodyid[i])
                mujoco.mj_jac(self.model, self.data, self.jac_b, None, contact.pos, self.model.geom_bodyid[j])
                grad[row] = contact.frame[:3] @ (self.jac_a - self.jac_b) @ self.mapping


class ConstrainedRetargeter:
    def __init__(self, solver, model, wuji=None, *, tsinghua=None):
        self.opt = solver.optimizer
        self.wuji = wuji
        self.tsinghua = tsinghua
        self.drive = self.opt.idx_pin2target
        self.robot = self.opt.robot
        self.last = self.robot.q0[self.drive].copy()
        self.limits = self.robot.joint_limits[self.drive]
        self.speed = self.robot.model.velocityLimit[self.drive]
        self.geom = MeshConstraints(model, self.opt.target_joint_names)
        self.palm = PalmDistances(model)
        self.palm_data = mujoco.MjData(model)
        self.exact_palm = False
        if not self.geom.feasible(self.last) or not self.palm.clear(self.geom.data, self.geom.clearance):
            raise ValueError("Neutral pose does not meet the collision clearance")
        self.minimizer = nlopt.opt(nlopt.LD_SLSQP, len(self.last))
        self.minimizer.add_inequality_mconstraint(self.geom.inequalities, [1e-8] * self.geom.count)
        self.minimizer.add_inequality_mconstraint(self.palm_inequalities, [1e-8] * len(self.palm.pairs))
        self.minimizer.set_ftol_abs(1e-6)
        self.minimizer.set_maxeval(40)
        self.held = 0

    def palm_inequalities(self, result, x, grad):
        if self.exact_palm:
            self.palm_data.qpos[:] = self.geom.mapping @ x
            mujoco.mj_kinematics(self.geom.model, self.palm_data)
            mujoco.mj_comPos(self.geom.model, self.palm_data)
            gaps, slopes = self.palm.linearize(self.palm_data, self.geom.mapping,
                                                overlap_slopes=self.palm_slopes)
            # Leave a micrometre for convergence at a curved mesh boundary.
            result[:] = self.geom.clearance + 1e-6 - gaps
            if grad.size:
                grad[:] = -slopes
            return
        result[:] = self.geom.clearance - self.palm_gaps - self.palm_slopes @ (x - self.last)
        if grad.size:
            grad[:] = -self.palm_slopes

    def full_pose(self, x):
        q = self.robot.q0.copy()
        q[self.drive] = x
        return self.opt.adaptor.forward_qpos(q)

    def solve(self, points, dt, *, aligned_points=None):
        if aligned_points is None:
            aligned_points = points
        origins, tips = self.opt.target_link_human_indices
        if self.tsinghua is not None:
            obj = self.tsinghua.callback(points, aligned_points, self.last)
        elif self.wuji is None:
            vectors = points[tips] - points[origins]
            # Keep inter-tip goals in the measured hand frame. Register the
            # palm and individual bone vectors to the GS S1 anatomy.
            anatomical = (origins == 0) | ((origins % 4) != 0)
            vectors[anatomical] = aligned_points[tips[anatomical]] - aligned_points[origins[anatomical]]
            obj = self.opt.get_objective_function(vectors.astype(np.float32),
                                                  np.zeros(len(self.opt.idx_pin2fixed)), self.last)
        else:
            w = self.wuji
            full_obj = w._get_objective_analytical(
                w._compute_tip_vectors(points, w.scaling), w._compute_tip_dirs(points),
                w._compute_full_hand_vectors(aligned_points, w.segment_scaling),
                w._compute_pinch_alpha(points), self.full_pose(self.last))
            def obj(x, grad):
                full_grad = np.empty(self.robot.dof)
                cost = full_obj(self.full_pose(x), full_grad)
                if grad.size:
                    grad[:] = self.opt.adaptor.backward_jacobian(full_grad)
                return cost

        return self.solve_objective(obj, dt)

    def solve_objective(self, obj, dt, *, initial=None):
        """Minimize a retargeting objective under the same GS S1 constraints."""
        self.geom.update(self.last)
        self.palm_gaps, self.palm_slopes = self.palm.linearize(self.geom.data, self.geom.mapping)
        self.exact_palm = False
        step = self.speed * dt
        self.minimizer.set_min_objective(obj)
        # A plane tangent to the palm gap can repeatedly admit an infeasible
        # thumb pose, even for a small step. Re-evaluate the actual gap inside
        # SLSQP on retry so unrelated finger joints can keep moving.
        for radius in (step, step * .5, step * .25):
            lower = np.maximum(self.limits[:, 0], self.last - radius)
            upper = np.minimum(self.limits[:, 1], self.last + radius)
            self.minimizer.set_lower_bounds(lower)
            self.minimizer.set_upper_bounds(upper)
            # SLSQP accepts a seed outside nonlinear constraints. Clipping a
            # nearby reference to velocity limits can introduce a tiny mesh
            # overlap. Let the optimizer repair it before checking the command.
            start = self.last if initial is None else np.clip(initial, lower, upper)
            try:
                candidate = self.minimizer.optimize(start)
            except nlopt.RoundoffLimited:
                continue
            if not np.isfinite(candidate).all() or not self.geom.feasible(candidate):
                continue
            if self.palm.clear(self.geom.data, self.geom.clearance - 1e-7):
                self.last = candidate.copy()
                break
            self.exact_palm = True
        else:
            self.held += 1
        return self.full_pose(self.last)


def load_wuji(source, assets, side, joint_names):
    import sys
    import yaml

    if not (source / "wuji_retargeting/opt/adaptive_analytical.py").is_file():
        raise FileNotFoundError(f"Wuji retargeting checkout not found at {source}")
    sys.path.insert(0, str(source))
    from wuji_retargeting.opt.adaptive_analytical import AdaptiveOptimizerAnalytical

    cfg = yaml.safe_load((source / "example/config/adaptive_analytical_video.yaml").read_text())
    cfg["optimizer"].update(hand_side=side, urdf_path=str(assets / f"hand/source/{side}_dexhand.urdf"),
                            link_naming=dict(palm="palm", fingers=["th", "ff", "mf", "rf", "lf"],
                                             pip="{finger}middle", dip="{finger}distal",
                                             tip="{finger}tip", link1="{finger}base"))
    # Wuji's tuned human-to-robot lengths are specific to its own CAD.
    cfg["retarget"]["segment_scaling"] = {}
    optimizer = AdaptiveOptimizerAnalytical(cfg)
    optimizer.set_timing_enabled(False)
    if optimizer.robot.dof_joint_names != joint_names:
        raise ValueError("Wuji and DexPilot robot models disagree on joint order")
    return optimizer
