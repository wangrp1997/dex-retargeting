"""Drive an upright Botyard GS S1 V3.2 hand from the preview's ROS targets."""

import argparse
import json
from functools import lru_cache
from pathlib import Path
import subprocess
from time import monotonic, sleep
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from gs_s1_palm import PalmDistances
from gs_s1_mano_overlay import ManoOverlay, add_overlay_asset


@lru_cache(maxsize=2)
def hand_urdf(side):
    share = Path(get_package_share_directory("dexhand_description"))
    return subprocess.check_output([
        "xacro", str(share / "urdf/dexhand.urdf.xacro"), "dexhand_type:=gs_s1",
        "sub_version:=2", f"use_left_hand:={str(side == 'left').lower()}", "use_ros2_control:=false",
        "use_gazebo:=false",
    ])


def hand_scene_xml(assets, mesh_dir, side):
    root = ET.parse(assets / "hand/source" / f"{side}_dexhand.xml").getroot()
    urdf = ET.fromstring(hand_urdf(side))
    ranges = {joint.get("name"): " ".join(joint.find("limit").attrib[k] for k in ("lower", "upper"))
              for joint in urdf.findall("joint[@type='revolute']")}
    profile = ET.parse(assets / "grasp/fivefinger_uniform48p8_side2deg_v1/scene.xml").getroot()
    root.find("compiler").attrib.pop("meshdir")
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str((mesh_dir / mesh.get("file")).resolve(strict=True)))
    for tag in ("option", "statistic", "visual"):
        root.append(profile.find(tag))
    asset = root.find("asset")
    for item in profile.find("asset"):
        if item.tag in ("texture", "material"):
            asset.append(item)
    checker = asset.find("texture[@name='groundplane']")
    checker.set("rgb1", ".32 .37 .42")
    checker.set("rgb2", ".14 .18 .22")
    asset.find("material[@name='groundplane']").set("texrepeat", "7 7")
    root.find("statistic").set("center", "0 0 .22")

    world = root.find("worldbody")
    base = world.find("body[@name='fabase']")
    base.attrib.pop("euler")  # upright mounting, base mesh begins at z=0
    floor = profile.find("worldbody/geom[@name='floor_collision']")
    floor.set("pos", "0 0 0")
    world.append(floor)
    world.extend(profile.findall("worldbody/light"))

    # Use the contact and servo settings of the tested five-finger scene on
    # each hand's own CAD, joint axes, equality constraints and exclusions.
    reference_geoms = {g.get("mesh"): g for g in profile.findall(".//geom[@mesh]")}
    for geom in base.iter("geom"):
        reference = reference_geoms[geom.get("mesh")]
        for attr in ("friction", "solref"):
            geom.set(attr, reference.get(attr))
    actuators = root.find("actuator")
    for body in base.iter("body"):
        joint = body.find("joint")
        if joint is None:
            continue
        name = joint.get("name")
        if name in ("FAJ1", "FAJ3"):
            body.remove(joint)
            actuators.remove(actuators.find(f"*[@joint='{name}']"))
        else:
            joint.set("range", ranges[name])
            reference = profile.find(f".//worldbody//joint[@name='{name}']")
            joint.set("actuatorfrcrange", reference.get("actuatorfrcrange"))
    for actuator in actuators:
        actuator.set("ctrlrange", ranges[actuator.get("joint")])
        reference = profile.find(f"actuator/*[@joint='{actuator.get('joint')}']")
        actuator.set("forcerange", reference.get("forcerange"))
    # The grasp scene excludes some neighboring finger bases. Free hand
    # motion must check every pair of different fingers, including the bases.
    fingers = {"th", "ff", "mf", "rf", "lf"}
    contact = root.find("contact")
    for pair in list(contact):
        a, b = pair.get("body1")[:2], pair.get("body2")[:2]
        if a in fingers and b in fingers and a != b:
            contact.remove(pair)
    return root


def load_hand(assets, mesh_dir, side, overlay_template=None):
    root = hand_scene_xml(assets, mesh_dir, side)
    if overlay_template is not None:
        add_overlay_asset(root, overlay_template)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.geom_margin[:] = .001  # engage soft contact before the mesh surfaces intersect
    return model


class JointStops:
    def __init__(self, model):
        # The upright GS S1 has only finger hinges. Use the compiled addresses so
        # the four passive joints receive the same stops as the actuated joints.
        hinges = np.flatnonzero(model.jnt_limited & (model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
        self.stop_positions = model.jnt_qposadr[hinges]
        self.stop_velocities = model.jnt_dofadr[hinges]
        self.lower, self.upper = model.jnt_range[hinges].T
    def apply(self, data):
        angles = data.qpos[self.stop_positions]
        below, above = angles < self.lower, angles > self.upper
        hit = below | above
        if np.any(hit):
            data.qpos[self.stop_positions] = np.clip(angles, self.lower, self.upper)
            velocities = data.qvel[self.stop_velocities]
            outward = (below & (velocities < 0)) | (above & (velocities > 0))
            data.qvel[self.stop_velocities[outward]] = 0
            data.qacc_warmstart[self.stop_velocities[hit]] = 0


class ContactStepper:
    """Advance the hand within the URDF stops and check contact at the resulting pose."""

    def __init__(self, model):
        self.model = model
        self.stops = JointStops(model)
        self.data = mujoco.MjData(model)
        self.probe = mujoco.MjData(model)
        self.palm = PalmDistances(model)
        mujoco.mj_forward(model, self.data)
        if np.any(self.data.contact.dist < 0) or not self.palm.clear(self.data, 0):
            raise ValueError("Initial hand configuration intersects a collision mesh")
        self.rejected = 0

    def clear_pose(self, qpos):
        self.probe.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.probe)
        mujoco.mj_comPos(self.model, self.probe)
        mujoco.mj_collision(self.model, self.probe)
        return not np.any(self.probe.contact.dist < 0) and self.palm.clear(self.probe, 0)

    def advance(self, steps):
        model, data = self.model, self.data
        for _ in range(steps):
            previous = data.qpos.copy()
            mujoco.mj_step(model, data)
            if np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all():
                self.stops.apply(data)
                # Probe the stopped pose separately; mj_collision resets the constraint arena.
                if self.clear_pose(data.qpos):
                    continue
            data.qpos[:] = previous
            data.qvel[:] = 0
            data.qacc_warmstart[:] = 0
            mujoco.mj_forward(model, data)
            self.rejected += 1


class HandSimulation:
    def __init__(self, node, model):
        self.node = node
        self.model = model
        self.physics = ContactStepper(model)
        self.data = self.physics.data
        self.command_stamp_ns = None
        self.drives = [model.joint(j).name for j in model.actuator_trnid[:, 0]]
        passive = {f"{finger}J1" for finger in ("FF", "MF", "RF", "LF")}
        joints = [model.joint(i).name for i in range(model.njnt)]
        if len(self.drives) != 16 or set(joints) - set(self.drives) != passive:
            raise ValueError("Unexpected GS S1 finger actuators or passive joints")
        self.qpos_indices = model.jnt_qposadr.copy()
        self.state = JointState()
        self.state.name = joints
        self.publisher = node.create_publisher(JointState, "sim_joint_states", 1)
        node.create_subscription(JointState, "joint_states", self.receive_target, 1)
        mujoco.mj_forward(model, self.data)

    def receive_target(self, msg):
        values = dict(zip(msg.name, msg.position))
        if not all(name in values for name in self.drives):
            return
        target = [values[name] for name in self.drives]
        if np.isfinite(target).all():
            self.data.ctrl[:] = np.clip(target, *self.model.actuator_ctrlrange.T)
            stamp = msg.header.stamp
            self.command_stamp_ns = stamp.sec * 10**9 + stamp.nanosec

    def advance(self, steps):
        self.physics.advance(steps)
        self.state.header.stamp = self.node.get_clock().now().to_msg()
        self.state.position = self.data.qpos[self.qpos_indices].tolist()
        self.publisher.publish(self.state)


def publish_recording_state(node, publisher, viewer, display, hand, overlay_pose):
    if not publisher.get_subscription_count():
        return
    stamp = node.get_clock().now().nanoseconds
    with viewer.lock():
        cam = viewer.cam
        camera = {name: getattr(cam, name) for name in ('distance', 'azimuth', 'elevation',
                                                         'fixedcamid', 'trackbodyid', 'type')}
        camera['lookat'] = cam.lookat.tolist()
        if hasattr(cam, 'orthographic'):
            camera['orthographic'] = cam.orthographic
        options = {key: getattr(viewer.opt, key).tolist() for key in ('geomgroup', 'sitegroup', 'flags')}
    # Capture the state that was just drawn, including all four passive J1s.
    state = dict(stamp_ns=stamp, qpos=display.qpos.tolist(), camera=camera, options=options,
                 applied_command_stamp_ns=hand.command_stamp_ns, ctrl_joint_names=hand.drives,
                 ctrl_rad=display.ctrl.tolist(), rejected_steps_total=hand.physics.rejected)
    if overlay_pose:
        state['mano_overlay_pose'] = overlay_pose
    publisher.publish(String(data=json.dumps(state, separators=(',', ':'))))


def run_viewer(node, model, side, record_view=False, mano_overlay=False):
    hand = HandSimulation(node, model)
    display = mujoco.MjData(model)
    mujoco.mj_forward(model, display)
    publisher = node.create_publisher(String, 'record_state', 1) if record_view else None
    overlay = ManoOverlay(model) if mano_overlay else None
    enabled = False
    latest = uploaded = ''

    def receive_pose(msg):
        nonlocal latest
        latest = msg.data

    def key_pressed(key):
        nonlocal enabled
        if key == ord('N'):
            enabled = not enabled

    if overlay:
        node.create_subscription(String, 'mano_overlay_pose', receive_pose, 1)
    with mujoco.viewer.launch_passive(model, display, key_callback=key_pressed if overlay else None, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0, 0, .23]
            viewer.cam.distance = .76
            viewer.cam.azimuth = -65 if side == "right" else -115
            viewer.cam.elevation = -12
        node.get_logger().info(f"MuJoCo {side} hand ready: 16 finger actuators, 4 passive J1 joints, fixed wrist")
        # Advance the physical clock in fixed steps. Refresh the display near 60 Hz.
        steps = max(1, round(1 / (60 * model.opt.timestep)))
        period = steps * model.opt.timestep
        deadline = monotonic()
        label = None
        while rclpy.ok() and viewer.is_running():
            rclpy.spin_once(node, timeout_sec=0.0)
            hand.advance(steps)
            visible = enabled
            pose = latest if visible else ''
            refresh = bool(pose and pose != uploaded)
            with viewer.lock():
                display.time = hand.data.time
                display.qpos[:] = hand.data.qpos
                display.qvel[:] = hand.data.qvel
                display.ctrl[:] = hand.data.ctrl
                mujoco.mj_forward(model, display)
                if overlay:
                    overlay.set_visible(bool(pose))
                    viewer.user_scn.ngeom = 0
                    if refresh:
                        overlay.update_vertices(pose)
                    if pose:
                        overlay.draw(viewer.user_scn, display)
            if refresh:
                viewer.update_mesh(overlay.mesh_id)  # takes its own viewer lock
                uploaded = pose
            if overlay:
                status = 'cyan MANO / orange GS S1' if pose else ('waiting for hand' if visible else 'off')
                if status != label:
                    viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, f'N: MANO overlay | {status}', ''))
                    label = status
            viewer.sync()
            if publisher is not None:
                publish_recording_state(node, publisher, viewer, display, hand, pose)
            deadline = max(deadline + period, monotonic())
            sleep(max(0.0, deadline - monotonic()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--mesh-dir", required=True, type=Path)
    parser.add_argument("--hand", required=True, choices=("right", "left"))
    parser.add_argument("--record-view", action="store_true")
    parser.add_argument("--overlay-template", type=Path)
    args = parser.parse_args()
    model = load_hand(args.assets, args.mesh_dir, args.hand, args.overlay_template)
    rclpy.init(args=[])
    node = rclpy.create_node("mujoco_hand", namespace="/gs_s1_preview")
    try:
        run_viewer(node, model, args.hand, args.record_view, args.overlay_template is not None)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
