"""Hardware protocol regressions; no ROS graph or physical hand required."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'hardware_commands', Path(__file__).resolve().parents[1]
    / 'example/vector_retargeting/gs_s1_hardware_commands.py')
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
JOINTS = protocol.WRIST_JOINTS + protocol.FINGER_JOINTS


def relay():
    return protocol.PositionRelay(JOINTS, {n: (-2., 2., .8) for n in JOINTS})


def test_named_mapping_holds_actual_wrists_and_ramps_fingers():
    r = relay()
    # Preview wrist order is opposite to the default controller. Its zero
    # wrist targets must never overwrite the measured hardware wrist pose.
    source_names = ('FAJ1', 'FAJ3') + protocol.FINGER_JOINTS
    r.set_feedback(list(reversed(JOINTS)), list(reversed([.3, -.4] + [0.] * 16)), 10.)
    r.set_target(source_names, [0., 0.] + [1.] * 16, 10.)
    assert r.step(10.) == [.3, -.4] + [0.] * 16
    assert r.step(10.1) == pytest.approx([.3, -.4] + [.08] * 16)
    # The configured velocity bound is respected even after a stalled timer.
    r.set_feedback(JOINTS, [.3, -.4] + [.08] * 16, 10.4)
    r.set_target(source_names, [0., 0.] + [1.] * 16, 10.4)
    assert r.step(10.4) == pytest.approx([.3, -.4] + [.16] * 16)


def test_controller_joint_order_is_read_from_actual_configuration():
    order = tuple(reversed(JOINTS))
    r = protocol.PositionRelay(order, {n: (-2., 2., 1.) for n in order})
    r.set_feedback(JOINTS, [.3, -.4] + [0.] * 16, 1.)
    r.set_target(protocol.FINGER_JOINTS, [.1] * 16, 1.)
    assert r.step(1.) == list(reversed([.3, -.4] + [0.] * 16))


@pytest.mark.parametrize('missing', ['feedback', 'target'])
def test_no_output_until_both_feedback_and_tracked_target_exist(missing):
    r = relay()
    if missing != 'feedback':
        r.set_feedback(JOINTS, [0.] * 18, 1.)
    if missing != 'target':
        r.set_target(protocol.FINGER_JOINTS, [1.] * 16, 1.)
    assert r.step(1.) is None


def test_stale_target_stops_and_resumption_starts_at_new_feedback():
    r = relay()
    r.set_feedback(JOINTS, [.1, -.1] + [0.] * 16, 1.)
    r.set_target(protocol.FINGER_JOINTS, [1.] * 16, 1.)
    r.step(1.)
    assert r.step(1.6) is None
    r.set_feedback(JOINTS, [.3, -.3] + [.4] * 16, 2.)
    r.set_target(protocol.FINGER_JOINTS, [1.] * 16, 2.)
    assert r.step(2.) == [.3, -.3] + [.4] * 16


def test_stale_feedback_stops_even_with_fresh_targets():
    r = relay()
    r.set_feedback(JOINTS, [0.] * 18, 1.)
    r.set_target(protocol.FINGER_JOINTS, [1.] * 16, 2.1)
    assert r.step(2.1) is None


@pytest.mark.parametrize('bad', ['duplicate', 'missing', 'nan', 'out_of_range'])
def test_invalid_targets_clear_the_previous_target(bad):
    r = relay()
    r.set_feedback(JOINTS, [0.] * 18, 1.)
    r.set_target(protocol.FINGER_JOINTS, [.5] * 16, 1.)
    names, values = list(protocol.FINGER_JOINTS), [.5] * 16
    if bad == 'duplicate':
        names[-1] = names[0]
    elif bad == 'missing':
        names, values = names[:-1], values[:-1]
    elif bad == 'nan':
        values[-1] = float('nan')
    else:
        values[-1] = 3.
    with pytest.raises((ValueError, KeyError)):
        r.set_target(names, values, 1.1)
    assert r.step(1.1) is None
