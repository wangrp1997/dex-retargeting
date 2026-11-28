"""Exact STL distances for moving links inside concave palm cutouts."""

from pathlib import Path

import mujoco
import numpy as np
from pinocchio import hppfcl


def load_body_mesh(model, loader, body):
    mesh = model.geom_dataid[model.body_geomadr[body]]
    path = model.paths[model.mesh_pathadr[mesh]:].split(b"\0", 1)[0].decode()
    return loader.load(str(Path(path).resolve(strict=True)))


class PalmDistances:
    def __init__(self, model):
        self.model = model
        self.pairs = []
        bodies = {}
        loader = hppfcl.MeshLoader()
        for sig in model.exclude_signature:
            a, b = int(sig) >> 16, int(sig) & 65535
            if model.body_weldid[a] == model.body_weldid[b] or model.body_parentid[b] == a:
                continue
            if model.body(a).name != "palm":
                raise ValueError(f"Unexpected nonadjacent exclusion: {model.body(a).name}/{model.body(b).name}")
            self.pairs.append((a, b))
            for body in (a, b):
                if body not in bodies:
                    bodies[body] = load_body_mesh(model, loader, body)
        self.bodies = bodies
        self.mobile = np.array([b for _, b in self.pairs])
        self.radii = np.array([np.linalg.norm(bodies[b].vertices(), axis=1).max() for b in self.mobile])
        self.rotations = np.zeros((len(self.mobile), 9))
        self.positions = np.zeros((len(self.mobile), 3))
        self.distances = np.full(len(self.mobile), -np.inf)
        self.request = hppfcl.DistanceRequest()
        self.request.enable_nearest_points = True
        self.pose_cache = [None] * len(self.pairs)

    def measure(self, data, i):
        a, b = self.pairs[i]
        key = np.r_[data.xpos[[a, b]].ravel(), data.xmat[[a, b]].ravel()]
        cached = self.pose_cache[i]
        if cached is not None and np.array_equal(key, cached[0]):
            return cached[1:]
        ta = hppfcl.Transform3f(data.xmat[a].reshape(3, 3), data.xpos[a])
        tb = hppfcl.Transform3f(data.xmat[b].reshape(3, 3), data.xpos[b])
        result = hppfcl.DistanceResult()
        distance = hppfcl.distance(self.bodies[a], ta, self.bodies[b], tb, self.request, result)
        self.distances[i] = distance
        self.rotations[i] = data.xmat[b]
        self.positions[i] = data.xpos[b]
        self.pose_cache[i] = key, distance, result
        return distance, result

    def clear(self, data, margin):
        # The palm is fixed. Translation plus a rotation-norm bound times the
        # mesh radius bounds the displacement of every point since the last
        # exact query. A certified gap can therefore skip the triangle search.
        travel = np.linalg.norm(data.xpos[self.mobile] - self.positions, axis=1)
        travel += self.radii * np.linalg.norm(data.xmat[self.mobile] - self.rotations, axis=1)
        for i in np.flatnonzero(self.distances - travel <= margin):
            if self.measure(data, i)[0] <= margin:
                return False
        return True

    def linearize(self, data, mapping, *, overlap_slopes=None):
        jac = np.empty((3, self.model.nv))
        slopes = np.zeros((len(self.pairs), mapping.shape[1]))
        gaps = np.empty(len(self.pairs))
        for i, (_, b) in enumerate(self.pairs):
            gaps[i], result = self.measure(data, i)
            if gaps[i] <= 0:
                if overlap_slopes is None:
                    raise ValueError("Collision constraint warm start intersects the palm")
                # Intersecting triangles have no unique separation normal.
                # Mark the trial infeasible and guide it using the safe pose.
                gaps[i] = 0.
                slopes[i] = overlap_slopes[i]
                continue
            normal = (result.getNearestPoint2() - result.getNearestPoint1()) / gaps[i]
            mujoco.mj_jac(self.model, data, jac, None, result.getNearestPoint2(), b)
            slopes[i] = normal @ jac @ mapping
        return gaps, slopes
