"""Fresh camera/MANO preview with a separate Botyard optimization process."""
from contextlib import ExitStack
import json
from time import monotonic

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from gs_s1_filter import LandmarkFilter
from gs_s1_live_pipeline import LatestCamera, LiveRetargeting, SolverProcess
from gs_s1_mano_overlay import encode_pose
from gs_s1_rviz import FINGER_JOINTS, MANO_TO_GS_S1, WRIST_JOINTS


def source_stamp(node, captured_at):
    """Map host capture receipt to ROS time, preserving processing age."""
    return (node.get_clock().now() - Duration(seconds=max(0., monotonic() - captured_at))).to_msg()


def stream_live(node, capture, solver, mano, monitor, detector, show_camera, config, *, publish_timing=False):
    publisher = node.create_publisher(JointState, 'joint_states', 10)
    tracked_publisher = node.create_publisher(JointState, 'tracked_joint_targets', 1)
    overlay_pub = node.create_publisher(String, 'mano_overlay_pose', 1)
    timing_pub = node.create_publisher(String, 'pipeline_timing', 20) if publish_timing else None

    def timing(**fields):
        if timing_pub is not None:
            timing_pub.publish(String(data=json.dumps(fields, allow_nan=False, separators=(',', ':'))))
    message = JointState()
    message.name = WRIST_JOINTS + FINGER_JOINTS
    indices = [solver.joint_names.index(name) for name in message.name]
    drives = solver.optimizer.idx_pin2target
    q = solver.optimizer.robot.q0.copy()
    message.position = q[indices].tolist()
    pinky_j2 = solver.joint_names.index('LFJ2')
    filtered = LandmarkFilter(config.smoothing)
    previous_tracking, show_mano = None, True
    frames = tracked = 0
    vision_seconds = solve_seconds = 0.
    last_solve = command_time = None
    title = f'GS S1 - {config.hand} - Botyard'
    node.get_logger().info('Latest-frame camera/MANO preview; independent Botyard solver, no input backlog')

    with ExitStack() as cleanup:
        worker = SolverProcess(config)
        cleanup.callback(worker.close)
        camera = LatestCamera(capture)
        cleanup.callback(camera.close)
        lane = LiveRetargeting(worker, q[drives])
        while rclpy.ok():
            sample = camera.take()
            if sample is not None:
                start = monotonic()
                frame, timestamp = sample.image, sample.captured_at
                frames += 1
                _, points, landmarks, wrist_axes = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tracking = points is not None and np.isfinite(points).all()
                if tracking:
                    pixels = detector.parse_keypoint_2d(landmarks, frame.shape)
                    tracking = (np.isfinite(wrist_axes).all() and np.isfinite(pixels).all()
                                and np.linalg.norm(np.cross(points[5] - points[0], points[9] - points[0])) > 1e-8)
                detection_s = monotonic() - start
                lane.observe(tracking, timestamp)
                status = f'Tracking {config.hand} hand' if tracking else f'Waiting for {config.hand} hand'
                if tracking != previous_tracking:
                    node.get_logger().info(status if tracking else status + ', holding pose')
                    previous_tracking = tracking
                if tracking:
                    mano_start = monotonic()
                    camera_points = points @ detector.operator2mano.T @ wrist_axes.T
                    joints, vertices = mano.fit(camera_points, pixels, frame.shape[1::-1], timestamp)
                    target = filtered.update(points @ MANO_TO_GS_S1, timestamp)
                    tracked += 1
                    mano_s = monotonic() - mano_start
                else:
                    mano_s = None
                if timing_pub is not None:
                    timing(kind='observation', sequence=sample.sequence,
                           captured_at_steady_ns=int(timestamp * 1e9), tracking=bool(tracking),
                           detection_s=detection_s, mano_fit_s=mano_s,
                           capture_to_vision_s=monotonic() - timestamp)
                vision_seconds += monotonic() - start
                if overlay_pub.get_subscription_count() > 0:
                    overlay_pub.publish(String(data=encode_pose(joints, vertices) if tracking else ''))

            # Observe tracking loss before accepting an in-flight solution.
            reply = lane.accept(monotonic())
            if reply is not None:
                q = reply.q
                message.position = q[indices].tolist()
                message.header.stamp = source_stamp(node, reply.captured_at)
                tracked_publisher.publish(message)
                last_solve = reply.solve_seconds
                solve_seconds += last_solve
                command_time = reply.captured_at
                if timing_pub is not None:
                    timing(kind='solution', sequence=reply.sequence,
                           source_stamp_ns=message.header.stamp.sec * 10**9 + message.header.stamp.nanosec,
                           solve_s=reply.solve_seconds, capture_to_publish_s=monotonic() - reply.captured_at)
            if sample is not None or reply is not None:
                publisher.publish(message)
            if sample is not None:
                if tracking:
                    dispatched = lane.submit(sample.sequence, timestamp, joints, vertices, monotonic())
                    if dispatched and timing_pub is not None:
                        timing(kind='dispatch', sequence=sample.sequence,
                               capture_to_dispatch_s=monotonic() - timestamp, solve_dt_s=lane.pending.dt)
                # Held preview poses retain their original timestamp. The
                # hardware topic above publishes each accepted solve once.
                if show_camera:
                    visible = tracking and show_mano and mano.draw_overlay(frame)
                    lines = [(status, 28)]
                    if tracking:
                        detector.draw_skeleton_on_image(frame, landmarks, style='default')
                        pair = monitor.first_intersection(q) if monitor is not None else None
                        if pair:
                            lines[0] = (f'Mesh collision: {pair[0]} / {pair[1]}', 28)
                        bones = np.diff(target[17:21], axis=0)
                        cross = np.linalg.norm(np.cross(bones[:-1], bones[1:]), axis=1)
                        pip, dip = np.degrees(np.arctan2(cross, (bones[:-1] * bones[1:]).sum(axis=1)))
                        lines.append((f'Pinky PIP {pip:.0f} / DIP {dip:.0f} | LFJ2 target '
                                      f'{np.degrees(q[pinky_j2]):.0f} deg', 52))
                        gaps = np.linalg.norm(target[8::4] - target[4], axis=1) * 1000
                        lines.append(('Thumb gaps mm: ' + ' '.join(
                            f'{name} {gap:.0f}' for name, gap in zip('IMRL', gaps)), 76))
                    mesh_status = 'off' if not show_mano else ('visible' if visible else 'waiting')
                    label = f'M: MANO {mesh_status} | vision {1000 * vision_seconds / frames:.0f} ms avg'
                    if last_solve is not None:
                        label += f' | solve {1000 * last_solve:.0f} ms'
                    else:
                        label += ' | solver starting' if not worker.ready else ' | waiting for solve'
                    lines.append((label, frame.shape[0] - 40))
                    age = f'{1000 * (monotonic() - command_time):.0f} ms' if command_time else 'waiting'
                    lines.append((f'Camera age {1000 * (monotonic() - timestamp):.0f} ms | command age {age}',
                                  frame.shape[0] - 16))
                    for label, y in lines:
                        for color, width in [((0, 0, 0), 3), ((255, 255, 255), 1)]:
                            cv2.putText(frame, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                                        .5, color, width, cv2.LINE_AA)
                    cv2.imshow(title, frame)
            if show_camera:
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q'), ord('Q')):
                    break
                if key in (ord('m'), ord('M')):
                    show_mano = not show_mano
        node.get_logger().info(f'Camera/MANO tracked {tracked}/{frames} processed frames; '
                               f'accepted {lane.accepted} robot solutions, discarded {lane.discarded} stale results')
        if lane.accepted:
            node.get_logger().info(f'Botyard worker mean {1000 * solve_seconds / lane.accepted:.1f} ms/solve')
