"""Publish webcam retargeting targets for the selected Botyard GS S1 hand."""

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import JointState

from dex_retargeting.retargeting_config import RetargetingConfig
from gs_s1_calibration import FingerRootCalibration
from gs_s1_contact import CollisionMonitor
from gs_s1_filter import LandmarkFilter
from gs_s1_mujoco import hand_urdf, load_hand
from gs_s1_optimization import ConstrainedRetargeter, load_wuji
from gs_s1_tsinghua import TsinghuaObjective
from single_hand_detector import SingleHandDetector


FINGER_JOINTS = [
    "THJ4", "THJ3", "THJ2", "THJ1",
    "FFJ4", "FFJ3", "FFJ2", "MFJ4", "MFJ3", "MFJ2",
    "RFJ4", "RFJ3", "RFJ2", "LFJ4", "LFJ3", "LFJ2",
]
WRIST_JOINTS = ["FAJ1", "FAJ3"]
OBJECTIVES = ("dexpilot", "wuji", "vector", "tsinghua")
# The detector changes MANO frames with handedness. Its lateral component
# already flips on mirrored input, as does the GS S1 palm's X coordinate.
MANO_TO_GS_S1 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])


def vector_config():
    # Every pair has a Cartesian target on every frame. The last four pairs
    # reproduce observed thumb-to-finger gaps continuously.
    pairs = []
    for digit, first in zip(("th", "ff", "mf", "rf", "lf"), (1, 5, 9, 13, 17)):
        pairs.append(("palm", digit + "tip", 0, first + 3))
        links = [digit + part for part in ("proximal", "middle", "distal", "tip")]
        for segment in range(1 if digit == "th" else 0, 3):
            pairs.append((links[segment], links[segment + 1], first + segment, first + segment + 1))
        if digit != "th":
            pairs.append(("thtip", digit + "tip", 4, first + 3))
    origins, tasks, source, target = zip(*pairs)
    return dict(type="vector", target_origin_link_names=list(origins), target_task_link_names=list(tasks),
                target_link_human_indices=np.array([source, target]))


def build_solver(show_collisions=False, hand="right", objective="dexpilot"):
    config = vector_config() if objective == "vector" else dict(
        type="dexpilot", wrist_link_name="palm",
        finger_tip_link_names=["thtip", "fftip", "mftip", "rftip", "lftip"])
    return build_configured_solver(config, hand, show_collisions)


def build_configured_solver(config, hand, show_collisions=False):
    share = Path(get_package_share_directory("dexhand_description"))
    xml = hand_urdf(hand)
    model = ET.fromstring(xml)
    # DexPilot uses the URDF fingertip frames. The overlay uses the original meshes.
    for link in model.findall("link"):
        for geometry in link.findall("visual") + link.findall("collision"):
            link.remove(geometry)
    with TemporaryDirectory() as directory:
        path = Path(directory) / f"gs_s1_{hand}.urdf"
        ET.ElementTree(model).write(path, encoding="utf-8")
        solver = RetargetingConfig(
            **config, urdf_path=str(path),
            target_joint_names=FINGER_JOINTS, low_pass_alpha=-1,
        ).build()
    if set(solver.optimizer.fixed_joint_names) != set(WRIST_JOINTS):
        raise ValueError(f"Unexpected fixed joints: {solver.optimizer.fixed_joint_names}")
    solver.set_qpos(solver.optimizer.robot.q0)
    solver.optimizer.set_joint_limit(solver.joint_limits, epsilon=0.0)
    monitor = (CollisionMonitor(solver.optimizer.robot.model, xml.decode(), share.parent)
               if show_collisions else None)
    return solver, monitor


def build_constrained(solver, model, method, args):
    wuji = (load_wuji(args.wuji_root, args.assets, args.hand, solver.joint_names)
            if method == "wuji" else None)
    tsinghua = TsinghuaObjective(args.tsinghua_root, solver.optimizer) if method == "tsinghua" else None
    return ConstrainedRetargeter(solver, model, wuji, tsinghua=tsinghua)


def stream(node, capture, solver, retargeter, monitor, detector, is_video, show_camera, hand,
           smoothing, objective_name):
    publisher = node.create_publisher(JointState, "joint_states", 10)
    message = JointState()
    message.name = WRIST_JOINTS + FINGER_JOINTS  # robot_state_publisher expands mimic joints
    message.position = [0.0] * len(message.name)
    indices = [solver.joint_names.index(name) for name in message.name]
    pinky_j2 = solver.joint_names.index("LFJ2")
    fps = capture.get(cv2.CAP_PROP_FPS)
    interval = 1.0 / fps if is_video and np.isfinite(fps) and fps > 0 else 0.0
    landmark_filter = LandmarkFilter(smoothing)
    botyard = objective_name == "botyard"
    show_mano = botyard
    calibration = FingerRootCalibration(solver.optimizer.robot) if show_camera and not botyard else None
    last_frame = monotonic() - 1 / 30
    frames = tracked = 0
    previous_tracking = None
    display_name = {"dexpilot": "DexPilot", "wuji": "Wuji", "botyard": "Botyard"}.get(objective_name, objective_name.title())
    window_title = f"GS S1 - {hand} - {display_name}"
    solve_seconds = 0.
    while rclpy.ok():
        start = monotonic()
        ok, frame = capture.read()
        if not ok:
            if is_video:
                break
            raise RuntimeError("Camera stopped returning frames.")
        capture_time = monotonic()
        timestamp = frames * interval if interval else capture_time
        dt = interval or min(capture_time - last_frame, .1)
        last_frame = capture_time
        frames += 1
        _, points, landmarks, wrist_axes = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tracking = points is not None and np.isfinite(points).all()
        if botyard and tracking:
            pixels = detector.parse_keypoint_2d(landmarks, frame.shape)
            tracking = (np.isfinite(wrist_axes).all() and np.isfinite(pixels).all()
                        and np.linalg.norm(np.cross(points[5] - points[0], points[9] - points[0])) > 1e-8)
        local_points = points @ MANO_TO_GS_S1 if tracking else None
        if calibration is not None:
            calibration.observe(local_points)
        status = f"Waiting for {hand} hand"
        if tracking:
            target = landmark_filter.update(local_points, timestamp)
            solve_start = monotonic()
            if botyard:
                camera_points = points @ detector.operator2mano.T @ wrist_axes.T
                q = retargeter.solve_camera(camera_points, pixels, frame.shape[1::-1], timestamp, dt)
            else:
                aligned = calibration.apply(target) if calibration is not None else target
                q = retargeter.solve(target, dt, aligned_points=aligned)
            solve_seconds += monotonic() - solve_start
            if not np.isfinite(q).all():
                raise ValueError("Non-finite retargeted joint position")
            message.position = q[indices].tolist()
            status = f"Tracking {hand} hand"
            tracked += 1
        if tracking != previous_tracking:
            node.get_logger().info(status if tracking else f"{status}, holding pose")
            previous_tracking = tracking
        message.header.stamp = node.get_clock().now().to_msg()
        publisher.publish(message)
        if show_camera:
            lines = []
            mesh_visible = tracking and show_mano and retargeter.mano.draw_overlay(frame)
            if tracking:
                detector.draw_skeleton_on_image(frame, landmarks, style="default")
                pair = monitor.first_intersection(q) if monitor is not None else None
                if pair:
                    status = f"Mesh collision: {pair[0]} / {pair[1]}"
                bones = np.diff(target[17:21], axis=0)
                cross = np.linalg.norm(np.cross(bones[:-1], bones[1:]), axis=1)
                pip, dip = np.degrees(np.arctan2(cross, (bones[:-1] * bones[1:]).sum(axis=1)))
                lines.append((f"Pinky PIP {pip:.0f} / DIP {dip:.0f} | LFJ2 target "
                              f"{np.degrees(q[pinky_j2]):.0f} deg", 52))
                gaps = np.linalg.norm(target[8::4] - target[4], axis=1) * 1000
                gap_text = " ".join(f"{name} {gap:.0f}" for name, gap in zip("IMRL", gaps))
                if objective_name == "dexpilot":
                    active = [name for name, on in zip("IMRL", solver.optimizer.projected[:4]) if on]
                    gap_text += " | projected: " + (",".join(active) or "-")
                lines.append(("Thumb gaps mm: " + gap_text, 76))
            lines.append((status, 28))
            if calibration is not None:
                lines.append((calibration.prompt(), frame.shape[0] - 16))
            elif botyard:
                mesh_status = "off" if not show_mano else ("visible" if mesh_visible else "waiting")
                label = f"M: MANO mesh {mesh_status}"
                if tracked:
                    label += f" | MANO + solve {1000 * solve_seconds / tracked:.0f} ms avg"
                lines.append((label, frame.shape[0] - 16))
            for label, y in lines:
                for color, width in [((0, 0, 0), 3), ((255, 255, 255), 1)]:
                    cv2.putText(frame, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                                .55, color, width, cv2.LINE_AA)
            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord("q")):
                break
            if key in (ord("m"), ord("M")) and botyard:
                show_mano = not show_mano
            elif key == ord("c") and calibration is not None:
                calibration.start()
            elif key == ord("b") and calibration is not None:
                calibration.toggle()
        if interval:
            sleep(max(0.0, interval - (monotonic() - start)))
    node.get_logger().info(f"Tracked {hand} hand in {tracked}/{frames} frames")
    if botyard and tracked:
        node.get_logger().info(f"Botyard MANO + solve mean {1000 * solve_seconds / tracked:.1f} ms/frame")
    if is_video and rclpy.ok() and not tracked:
        raise RuntimeError(f"No {hand} hand detected in the video.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    projects = Path(__file__).resolve().parents[3]
    parser.add_argument("--objective", choices=(*OBJECTIVES, "botyard"), default="dexpilot")
    parser.add_argument("--botyard-gradient", choices=("finite_difference", "analytic"), default="finite_difference",
                        help="Gradient of the first-stage Botyard hand-shape objective")
    parser.add_argument("--smoothing", choices=("one_euro", "off"), default="one_euro")
    parser.add_argument("--assets", type=Path, default=projects / "tendon-spin/assets")
    parser.add_argument("--wuji-root", type=Path, default=projects / "wuji-retargeting")
    parser.add_argument("--tsinghua-root", type=Path, default=projects / "tsinghua-retargeting")
    parser.add_argument("--mano-model", type=Path, help="Prepared MANO NPZ for the selected hand")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    source.add_argument("--video", type=Path, help="Replay an MP4 instead of using the camera")
    parser.add_argument("--no-camera-window", action="store_true")
    parser.add_argument("--no-target-collision-overlay", action="store_true")
    args = parser.parse_args()
    show_camera = not args.no_camera_window
    solver, monitor = build_solver(show_collisions=show_camera and not args.no_target_collision_overlay,
                                  hand=args.hand, objective=args.objective)
    mesh_dir = Path(get_package_share_directory("dexhand_description")) / "meshes/gs_s1"
    model = load_hand(args.assets, mesh_dir, args.hand)
    if args.objective == "botyard":
        # Import the same two objectives as the offline benchmark, only when selected.
        sys.path.append(str(Path(__file__).resolve().parent / "data/gs_s1_hand_baselines"))
        from botyard_live import BotyardRetargeter
        retargeter = BotyardRetargeter(solver, model, mesh_dir, args.hand, args.mano_model, args.smoothing,
                                       gradient=args.botyard_gradient)
    else:
        retargeter = build_constrained(solver, model, args.objective, args)
    device = str(args.video) if args.video else args.camera
    with ExitStack() as cleanup:
        capture = cv2.VideoCapture(device)
        cleanup.callback(capture.release)
        if args.video is None:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if show_camera:
            cleanup.callback(cv2.destroyAllWindows)
        if not capture.isOpened():
            parser.exit(1, f"Could not open {device!r}. Check --camera or --video.\n")
        detector = SingleHandDetector(hand_type=args.hand.title(), selfie=False)
        cleanup.callback(detector.hand_detector.close)
        rclpy.init(args=[])
        cleanup.callback(rclpy.try_shutdown)
        node = rclpy.create_node("camera_retargeting", namespace="/gs_s1_preview")
        node.get_logger().info(f"GS S1 {args.hand} hand, {args.objective} objective, "
                               f"landmark smoothing {args.smoothing}, mesh constraints enabled")
        cleanup.callback(node.destroy_node)
        try:
            stream(node, capture, solver, retargeter, monitor, detector, args.video is not None,
                   show_camera, args.hand, args.smoothing, args.objective)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
