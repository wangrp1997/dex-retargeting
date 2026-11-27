"""Display the fitted MANO surface in a palm-width-matched GS S1 frame."""
import base64
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


MESH_NAME = 'botyard_mano_overlay'


def palm_frame(wrist, index, middle, little):
    radial = index - little
    width = np.linalg.norm(radial)
    if width < 1e-8:
        raise ValueError('Degenerate palm width')
    radial = radial / width
    longitudinal = middle - wrist
    front = np.cross(radial, longitudinal)
    if np.linalg.norm(front) < 1e-8:
        raise ValueError('Degenerate palm plane')
    front /= np.linalg.norm(front)
    return index, np.column_stack((radial, front, np.cross(front, radial))), width


def encode_pose(joints, vertices):
    origin, axes, width = palm_frame(*joints[[0, 5, 9, 17]])
    points = np.vstack((joints[4], vertices))
    normalized = (points - origin) @ axes / width
    return base64.b64encode(normalized.astype('<f4').tobytes()).decode('ascii')


def normals(vertices, faces):
    a, b, c = vertices[faces].transpose(1, 0, 2)
    face_normals = np.cross(b - a, c - a)
    result = np.zeros_like(vertices)
    for corner in faces.T:
        np.add.at(result, corner, face_normals)
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)


def add_overlay_asset(scene, template):
    # An unreferenced visual mesh asset adds no bodies, geoms or contacts.
    with np.load(template, allow_pickle=False) as mano:
        vertices, faces = mano['v_template'], mano['f']
    mesh = ET.SubElement(scene.find('asset'), 'mesh', name=MESH_NAME)
    for key, values in [('vertex', vertices), ('normal', normals(vertices, faces)), ('face', faces)]:
        mesh.set(key, ' '.join(map(str, values.ravel())))


class ManoOverlay:
    def __init__(self, model):
        self.mesh_id = model.mesh(MESH_NAME).id
        mesh = self.mesh_id
        start = model.mesh_vertadr[mesh]
        self.vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        start = model.mesh_normaladr[mesh]
        self.normals = model.mesh_normal[start:start + model.mesh_normalnum[mesh]]
        start = model.mesh_faceadr[mesh]
        self.faces = model.mesh_face[start:start + model.mesh_facenum[mesh]]
        if self.vertices.shape != self.normals.shape:
            raise ValueError('MANO display mesh needs one normal per vertex')
        neutral = mujoco.MjData(model)
        mujoco.mj_kinematics(model, neutral)
        self.rgba = model.geom_rgba
        finger_bodies = ('th', 'ff', 'mf', 'rf', 'lf')
        self.hand_geoms = [i for i, b in enumerate(model.geom_bodyid)
                           if model.body(b).name == 'palm' or model.body(b).name.startswith(finger_bodies)]
        self.solid_alpha = self.rgba[self.hand_geoms, 3].copy()
        self.palm = model.body('palm').id
        self.robot_thumb = model.body('thtip').id
        self.mano_thumb = None
        wrist = neutral.xpos[self.palm]
        rotation = neutral.xmat[self.palm].reshape(3, 3)
        roots = neutral.xpos[[model.body(name).id for name in ('ffproximal', 'mfproximal', 'lfproximal')]]
        self.origin, self.axes, self.width = palm_frame(np.zeros(3), *((roots - wrist) @ rotation))

    def set_visible(self, visible):
        # Reveal the MANO surface and tip markers when the robot is in front.
        self.rgba[self.hand_geoms, 3] = self.solid_alpha * (.55 if visible else 1.)

    def update_vertices(self, payload):
        points = np.frombuffer(base64.b64decode(payload, validate=True), dtype='<f4').reshape(-1, 3)
        if len(points) != len(self.vertices) + 1 or not np.isfinite(points).all():
            raise ValueError('Invalid MANO overlay coordinates')
        local = self.origin + self.width * points @ self.axes.T
        self.mano_thumb = local[0]
        self.vertices[:] = local[1:]
        self.normals[:] = normals(self.vertices, self.faces)

    def draw(self, scene, data):
        # Append to either the passive viewer's user scene or the exporter's scene.
        wrist = data.xpos[self.palm]
        rotation = data.xmat[self.palm].reshape(3, 3)
        tip = wrist + rotation @ self.mano_thumb
        robot_tip = data.xpos[self.robot_thumb]

        def geom(kind, size, pos, rgba):
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, kind, size, pos, rotation.ravel(), rgba)
            scene.ngeom += 1
            return g

        surface = geom(mujoco.mjtGeom.mjGEOM_MESH, [.1, .1, .1], wrist, [.12, .85, .92, .36])
        # MuJoCo allocates two render IDs per mesh (surface and convex hull).
        surface.dataid = 2 * self.mesh_id
        surface.modelrbound = .25
        surface.transparent = 1
        for point, color in [(tip, [.05, 1., 1., 1.]), (robot_tip, [1., .35, .035, 1.])]:
            marker = geom(mujoco.mjtGeom.mjGEOM_SPHERE, [.0045] * 3, point, color)
            marker.emission = .6
        if np.linalg.norm(tip - robot_tip) > 1e-6:
            line = geom(mujoco.mjtGeom.mjGEOM_LINE, [0., 0., 0.], tip, [1., .94, .65, 1.])
            mujoco.mjv_connector(line, mujoco.mjtGeom.mjGEOM_LINE, 2., tip, robot_tip)
