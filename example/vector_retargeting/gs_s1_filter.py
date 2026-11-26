"""Speed-adaptive filtering of wrist-local 3D landmarks in metres."""

import numpy as np


def _alpha(cutoff, dt):
    scale = 2 * np.pi * cutoff * dt
    return scale / (1 + scale)


class LandmarkFilter:
    """One Euro filter with a single cutoff for each landmark's three axes."""

    def __init__(self, mode="one_euro"):
        if mode not in ("one_euro", "off"):
            raise ValueError(f"Unknown landmark smoothing mode: {mode}")
        self.mode = mode
        self.last_time = None
        self.position = None
        self.velocity = None

    def update(self, points, timestamp):
        if self.mode == "off":
            return points
        dt = None if self.last_time is None else timestamp - self.last_time
        if dt is None or dt > 0.25 or dt < 0:
            # Seed from the current hand when it reappears after a tracking gap.
            self.position = points.copy()
            self.velocity = np.zeros_like(points)
        elif dt > 0:
            velocity = (points - self.position) / dt
            self.velocity += _alpha(1.0, dt) * (velocity - self.velocity)
            speed = np.linalg.norm(self.velocity, axis=1, keepdims=True)
            cutoff = 1.3 + 20.0 * speed  # Hz, with speed in metres per second
            self.position = self.position + _alpha(cutoff, dt) * (points - self.position)
        self.last_time = timestamp
        return self.position
