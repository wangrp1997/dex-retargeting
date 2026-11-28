"""Exercise real publish scheduling with fake camera/vision/solver endpoints."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_live_stream_publishes_fresh_solutions_once_and_keeps_source_stamps(monkeypatch):
    pytest.importorskip('rclpy')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'example/vector_retargeting'))
    import gs_s1_live as live
    from gs_s1_live_pipeline import CameraFrame, PoseResult, SolverConfig
    from rclpy.time import Time

    clock = SimpleNamespace(value=100., frame=-1)
    endpoints = {}

    class Camera:
        def __init__(self, capture):
            self.closed = False
            endpoints['camera'] = self

        def take(self):
            clock.value += 1 / 30
            clock.frame += 1
            return CameraFrame(clock.frame, clock.value, np.zeros((100, 100, 3), np.uint8))

        def close(self):
            self.closed = True

    class Worker:
        ready = True

        def __init__(self, config):
            self.task, self.countdown, self.closed = None, 0, False
            endpoints['worker'] = self

        def send(self, task):
            assert self.task is None
            self.task, self.countdown = task, 3

        def poll(self):
            if self.task is None:
                return None
            self.countdown -= 1
            if self.countdown:
                return None
            task, self.task = self.task, None
            return PoseResult(task.sequence, task.generation, task.captured_at,
                              np.r_[np.zeros(2), task.previous + .1], task.previous + .1, .1)

        def close(self):
            self.closed = True

    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(deepcopy(message))

        def get_subscription_count(self):
            return 1

    class Node:
        def create_publisher(self, kind, name, depth):
            endpoints[name] = Publisher()
            return endpoints[name]

        def get_clock(self):
            return SimpleNamespace(now=lambda: Time(seconds=1000. + clock.value))

        def get_logger(self):
            return SimpleNamespace(info=lambda *args: None)

    joints = np.zeros((21, 3))
    joints[[5, 9, 17]] = [[.03, .06, 0], [0, .07, 0], [-.03, .05, 0]]

    class Detector:
        operator2mano = np.eye(3)

        def detect(self, image):
            # Frame 0's result arrives at frame 3 during a tracking outage.
            return None, (None if clock.frame in (3, 4) else joints), None, np.eye(3)

        def parse_keypoint_2d(self, landmarks, shape):
            return np.zeros((21, 2))

    monkeypatch.setattr(live, 'LatestCamera', Camera)
    monkeypatch.setattr(live, 'SolverProcess', Worker)
    monkeypatch.setattr(live, 'monotonic', lambda: clock.value)
    monkeypatch.setattr(live.rclpy, 'ok', lambda: clock.frame < 11)
    solver = SimpleNamespace(joint_names=live.WRIST_JOINTS + live.FINGER_JOINTS,
                             optimizer=SimpleNamespace(idx_pin2target=np.arange(2, 18),
                                                        robot=SimpleNamespace(q0=np.zeros(18))))
    mano = SimpleNamespace(fit=lambda *args: (joints, np.zeros((778, 3))))
    live.stream_live(Node(), None, solver, mano, None, Detector(), False,
                     SolverConfig('', 'left', None, 'off', 'analytic', 'v4_2'))
    tracked = endpoints['tracked_joint_targets'].messages
    assert len(tracked) == 2  # The result from frame 0 was discarded.
    stamps = [m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec for m in tracked]
    assert stamps == pytest.approx([1100. + 6 / 30, 1100. + 9 / 30])
    assert all(stamp < 1100. + 12 / 30 for stamp in stamps)
    preview = endpoints['joint_states'].messages
    assert len(preview) == 12
    # A held pose does not acquire a newer command/source timestamp.
    assert preview[8].header.stamp == preview[9].header.stamp
    assert preview[9].header.stamp == preview[10].header.stamp
    overlays = endpoints['mano_overlay_pose'].messages
    assert overlays[3].data == overlays[4].data == ''
    assert overlays[5].data
    assert endpoints['camera'].closed and endpoints['worker'].closed


def test_hardware_timeout_includes_time_spent_solving(monkeypatch):
    pytest.importorskip('rclpy')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'example/vector_retargeting'))
    import gs_s1_hardware as hardware
    from gs_s1_hardware_commands import FINGER_JOINTS, PositionRelay, WRIST_JOINTS
    from rclpy.time import Time
    from sensor_msgs.msg import JointState

    names = WRIST_JOINTS + FINGER_JOINTS
    relay = PositionRelay(names, {name: (-2., 2., 1.) for name in names})
    relay.set_feedback(names, [0.] * 18, 10.)
    message = JointState()
    message.name, message.position = list(FINGER_JOINTS), [.5] * 16
    message.header.stamp = Time(seconds=100.).to_msg()
    node = SimpleNamespace(relay=relay,
                           get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=100.4)))
    monkeypatch.setattr(hardware, 'monotonic', lambda: 10.)
    hardware.HardwareRelay.target(node, message)
    assert relay.target_time == pytest.approx(9.6)
    assert relay.step(10.) is not None
    assert relay.step(10.11) is None  # Not another .5 seconds from receipt.
