"""Read-only self-collision diagnostics for the GS S1 camera preview."""

from itertools import combinations

import pinocchio as pin


class CollisionMonitor:
    def __init__(self, model, urdf_xml, package_dir):
        self.model = model
        self.data = model.createData()
        self.geometry = pin.buildGeomFromUrdfString(
            model, urdf_xml, pin.GeometryType.COLLISION, package_dirs=str(package_dir)
        )
        objects = self.geometry.geometryObjects
        self.pair_names = []
        for i, j in combinations(range(len(objects)), 2):
            a, b = objects[i].parentJoint, objects[j].parentJoint
            # Same rigid body or adjacent joint housings can overlap by design.
            if a == b or a == model.parents[b] or b == model.parents[a]:
                continue
            self.geometry.addCollisionPair(pin.CollisionPair(i, j))
            self.pair_names.append((model.frames[objects[i].parentFrame].name,
                                    model.frames[objects[j].parentFrame].name))
        self.geom_data = self.geometry.createData()

    def first_intersection(self, q):
        if pin.computeCollisions(self.model, self.data, self.geometry, self.geom_data, q, True):
            for pair, result in zip(self.pair_names, self.geom_data.collisionResults):
                if result.isCollision():
                    return pair
        return None
