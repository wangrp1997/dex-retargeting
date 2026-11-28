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


def test_stale_target_stops_and_resumption_preserves_last_command():
    r = relay()
    r.set_feedback(JOINTS, [.1, -.1] + [0.] * 16, 1.)
    r.set_target(protocol.FINGER_JOINTS, [1.] * 16, 1.)
    r.step(1.)
    assert r.step(1.1) == pytest.approx([.1, -.1] + [.08] * 16)
    assert r.step(1.2) == pytest.approx([.1, -.1] + [.16] * 16)
    assert r.step(1.6) is None
    # The hand's sensor position lags the command. A new target resumes from
    # .16, rather than rewinding to that delayed .04 measurement.
    r.set_feedback(JOINTS, [.3, -.3] + [.04] * 16, 2.)
    r.set_target(protocol.FINGER_JOINTS, [1.] * 16, 2.)
    assert r.step(2.) == pytest.approx([.1, -.1] + [.24] * 16)


def test_feedback_loss_restarts_from_new_measurement():
    r = relay()
    r.set_feedback(JOINTS, [.1, -.1] + [0.] * 16, 1.)
    r.set_target(protocol.FINGER_JOINTS, [1.] * 16, 1.)
    r.step(1.)
    r.step(1.1)
    r.set_target(protocol.FINGER_JOINTS, [1.] * 16, 2.2)
    assert r.step(2.2) is None
    assert r.command is None
    r.set_feedback(JOINTS, [.3, -.3] + [.4] * 16, 2.3)
    assert r.step(2.3) == [.3, -.3] + [.4] * 16


def test_controller_reset_restarts_from_new_measurement():
    r = relay()
    r.set_feedback(JOINTS, [0.] * 18, 1.)
    r.set_target(protocol.FINGER_JOINTS, [1.] * 16, 1.)
    r.step(1.)
    r.step(1.1)
    r.reset()  # HardwareRelay calls this while controller status is invalid.
    r.set_feedback(JOINTS, [.3, -.3] + [.4] * 16, 1.2)
    assert r.step(1.2) == [.3, -.3] + [.4] * 16


def test_half_second_feedback_delay_does_not_rewind_repeated_resumptions():
    r = relay()
    history = [(0., [0.] * 18)]
    previous, previous_time = None, None
    pauses = 0
    for tick in range(200):
        now = tick * .02
        delayed = next((values for stamp, values in reversed(history) if stamp <= now - .5), [0.] * 18)
        r.set_feedback(JOINTS, delayed, now)
        if tick % 18 == 0:
            # Computed targets arrive 300 ms after their source observation.
            r.set_target(protocol.FINGER_JOINTS, [1.] * 16, now - .3)
        output = r.step(now)
        if output is None:
            pauses += 1
            continue
        if previous is not None:
            assert output[2] >= previous[2] - 1e-10
            assert output[2] - previous[2] <= .8 * min(.1, now - previous_time) + 1e-10
        previous, previous_time = output, now
        history.append((now, output))
    assert pauses > 0
    assert previous[2:] == pytest.approx([1.] * 16)


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
