"""Map four observed finger chains into the GS S1 palm frame and proportions."""

import numpy as np
from scipy.spatial.transform import Rotation


class FingerRootCalibration:
    sample_count = 20
    knuckles = [5, 9, 13, 17]  # MediaPipe MCP landmarks, index through little finger

    def __init__(self, robot):
        robot.compute_forward_kinematics(robot.q0)
        palm = robot.get_link_pose_inv(robot.get_link_index("palm"))
        self.robot_chains = np.array([
            [(palm @ robot.get_link_pose(robot.get_link_index(digit + link)))[:3, 3]
             for link in ("proximal", "middle", "distal", "tip")]
            for digit in ("ff", "mf", "rf", "lf")
        ])
        self.robot_bones = np.diff(self.robot_chains, axis=1)
        self.lengths = np.linalg.norm(self.robot_bones, axis=2, keepdims=True)
        self.samples = None
        self.alignment = None
        self.enabled = False

    def start(self):
        self.samples = []

    def toggle(self):
        if self.alignment is not None:
            self.enabled = not self.enabled
        self.samples = None

    def observe(self, points):
        if self.samples is None:
            return
        if points is None:
            self.samples.clear()
            return
        fingers = (points[5:] - points[0]).reshape(4, 4, 3)
        bones = np.diff(fingers, axis=1)
        lengths = np.linalg.norm(bones, axis=2)
        reach = np.linalg.norm(fingers[:, -1] - fingers[:, 0], axis=1)
        # A curled or collapsed finger cannot define the straight reference.
        # This check only applies during the requested calibration capture.
        if np.any(lengths < 1e-6) or np.any(reach < .85 * lengths.sum(axis=1)):
            self.samples.clear()
            return
        self.samples.append(fingers)
        if len(self.samples) == self.sample_count:
            rest = np.median(self.samples, axis=0)
            rest_bones = np.diff(rest, axis=1)
            rest_lengths = np.linalg.norm(rest_bones, axis=2, keepdims=True)
            if np.any(rest_lengths < 1e-6):
                self.samples.clear()
                return
            self.human_directions = rest_bones / rest_lengths
            self.human_roots = rest[:, 0]
            # One fixed rotation per phalanx aligns its observed straight direction
            # with the corresponding neutral CAD segment, including the fingertip site.
            self.alignment = Rotation.concatenate([
                Rotation.align_vectors([robot_dir], [human_dir])[0]
                for robot_dir, human_dir in zip(
                    (self.robot_bones / self.lengths).reshape(-1, 3),
                    self.human_directions.reshape(-1, 3))
            ])
            self.enabled = True
            self.samples = None

    def apply(self, points):
        if not self.enabled:
            return points
        fingers = points[5:].reshape(4, 4, 3)
        bones = np.diff(fingers, axis=1)
        lengths = np.linalg.norm(bones, axis=2, keepdims=True)
        directions = np.divide(bones, lengths, out=self.human_directions.copy(), where=lengths > 1e-6)
        robot_directions = self.alignment.apply(directions.reshape(-1, 3)).reshape(4, 3, 3)
        roots = self.robot_chains[:, 0] + (fingers[:, 0] - self.human_roots)
        aligned = points.copy()
        chains = aligned[5:].reshape(4, 4, 3)
        chains[:, 0] = roots
        chains[:, 1:] = roots[:, None, :] + np.cumsum(self.lengths * robot_directions, axis=1)
        return aligned

    def prompt(self):
        if self.samples is not None:
            return f"Hold 4 fingers straight and together: {len(self.samples)}/{self.sample_count}"
        if self.alignment is None:
            return "C: calibrate straight fingers"
        state = "ON" if self.enabled else "OFF"
        return f"Finger calibration {state} | B: compare | C: recalibrate"
