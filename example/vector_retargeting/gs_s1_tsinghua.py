"""Fixed-wrist specialization of Mingrui-Yu/retargeting's five-finger objective."""

from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml
from scipy.special import expit


class TsinghuaObjective:
    def __init__(self, source, optimizer):
        profile_file = Path(source) / "configs/retargeting_profiles/vector_wrist_joint_panda_shadow.yaml"
        if not profile_file.is_file():
            raise FileNotFoundError(f"Tsinghua five-finger profile missing at {profile_file}")
        profile = yaml.safe_load(profile_file.read_text())
        method = yaml.safe_load((source / profile["method"]).read_text())
        embodiment = yaml.safe_load((source / profile["robot"]).read_text())
        if method["optimizer"]["class"] != "VectorWristJointOptimizerV2" or method["ablation_option"] != 0:
            raise ValueError("Expected the full VectorWristJointOptimizerV2 method")
        self.beta = method["optimizer"]["params"]["huber_delta"]
        self.objective = profile["objective"]
        self.robot = optimizer.robot
        self.adaptor = optimizer.adaptor
        self.drive = optimizer.idx_pin2target
        self.rest = self.robot.q0[self.drive].copy()

        tips = embodiment["benchmark"]["fingertips"]
        self.human_tips = [tip["human_tip_index"] for tip in tips]
        self.human_distals = [tip["human_direction_base_index"] for tip in tips]
        self.wrist_index = profile["retargeting"]["human_wrist_index"]
        tip_links = [tip["link_name"] for tip in tips]
        wrist = profile["target"]["wrist_link_name"]
        pairs = profile["target"]["link_pairs"]
        expected = ([["world", tip_links[0]]] + [[wrist, tip] for tip in tip_links]
                    + [[tip_links[0], tip] for tip in tip_links[1:]]
                    + [[tip.removesuffix("tip") + "distal", tip] for tip in tip_links])
        if not tips[0]["is_thumb"] or len(tips) != 5 or pairs != expected:
            raise ValueError("Expected world/thumb, wrist/tips, thumb/tips and distal/tips in order")
        # In a fixed palm frame, world-to-thumb is also palm-to-thumb.
        pairs = [["palm" if link in ("world", wrist) else link for link in pair] for pair in pairs]
        names = list(dict.fromkeys(link for pair in pairs for link in pair))
        self.frames = [self.robot.get_link_index(name) for name in names]
        self.origins, self.tasks = np.array([[names.index(link) for link in pair] for pair in pairs]).T
        self.robot.compute_forward_kinematics(self.robot.q0)
        self.palm_rotation = self.robot.get_link_pose(self.robot.get_link_index("palm"))[:3, :3].copy()

        indices = [embodiment["actuated_joints"].index(name) for name in optimizer.target_joint_names]
        self.position_weight = np.asarray(profile["retargeting"]["joint_position_weights"])[indices]
        self.velocity_weight = np.asarray(profile["retargeting"]["joint_velocity_weights"])[indices]

    def reference(self, points, aligned):
        tips = points[self.human_tips]
        between = tips[1:] - tips[0]
        gap = np.linalg.norm(between, axis=1)
        transition = self.objective["pinch_transition_threshold"]
        contact = self.objective["pinch_contact_threshold"]
        slope = self.objective["pinch_sigmoid_slope"]
        scale = np.clip((gap - contact) / (transition - contact), 0.0, 1.0)
        target_gap = np.where(gap >= transition, gap, transition * scale)
        ratio = np.divide(target_gap, gap, out=np.zeros_like(gap), where=gap > 0)
        aligned_tips = aligned[self.human_tips]
        palm_to_tips = aligned_tips - aligned[self.wrist_index]
        targets = np.vstack((palm_to_tips[0], palm_to_tips, between * ratio[:, None],
                             aligned_tips - aligned[self.human_distals]))
        w = self.objective["weights"]
        weights = np.r_[w["world_thumb"],
                        w["wrist_fingertip"] * expit(slope * (np.r_[gap.min(), gap] - transition)),
                        w["thumb_primary"] * expit(slope * (transition - gap)),
                        np.full(len(tips), w["fingertip_orientation"])]
        return targets @ self.palm_rotation.T, weights

    def smooth_l1(self, residual):
        # Same mean reduction and beta as the upstream torch.nn.SmoothL1Loss.
        mag = np.abs(residual)
        cost = np.where(mag < self.beta, .5 * residual**2 / self.beta, mag - .5 * self.beta)
        return cost.mean(), np.clip(residual / self.beta, -1., 1.) / residual.size

    def callback(self, points, aligned, previous):
        targets, weights = self.reference(points, aligned)
        previous = previous.copy()
        model, data = self.robot.model, self.robot.data

        def objective(x, grad):
            q = self.robot.q0.copy()
            q[self.drive] = x
            self.adaptor.forward_qpos(q)
            if grad.size:
                pin.computeJointJacobians(model, data, q)
            else:
                pin.forwardKinematics(model, data, q)
            xyz = np.array([pin.updateFramePlacement(model, data, f).translation for f in self.frames])
            delta = xyz[self.tasks] - xyz[self.origins] - targets
            dist = np.linalg.norm(delta, axis=1)
            cartesian, d_cart = self.smooth_l1(weights * dist)
            position, d_pos = self.smooth_l1(self.position_weight * (x - self.rest))
            velocity, d_vel = self.smooth_l1(self.velocity_weight * (x - previous))
            if grad.size:
                jac = np.stack([pin.getFrameJacobian(model, data, f, pin.LOCAL_WORLD_ALIGNED)[:3]
                                for f in self.frames])
                direction = np.divide(delta, dist[:, None], out=np.zeros_like(delta),
                                      where=dist[:, None] > 0)
                d_xyz = (d_cart * weights)[:, None] * direction
                d_q = np.einsum("ij,ijk->k", d_xyz, jac[self.tasks] - jac[self.origins])
                grad[:] = (self.adaptor.backward_jacobian(d_q) + self.position_weight * d_pos
                           + self.velocity_weight * d_vel)
            return cartesian + position + velocity

        return objective
