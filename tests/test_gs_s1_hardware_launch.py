"""Inspect launch actions without opening a camera, serial port or ROS graph."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def launch_module(monkeypatch, tmp_path):
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path / 'ros_logs'))
    pytest.importorskip('launch')
    pytest.importorskip('launch_ros')
    spec = importlib.util.spec_from_file_location(
        'preview_launch', Path(__file__).resolve().parents[1]
        / 'example/vector_retargeting/gs_s1_preview.launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context(module, backend, **overrides):
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    result = LaunchContext()
    result.launch_configurations.update(backend=backend, objective='botyard', **overrides)
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(result)
    return result


@pytest.mark.parametrize(('backend', 'expected'), [('hardware', 'left'), ('mujoco', 'right'), ('rviz', 'right')])
def test_backend_hand_defaults(launch_module, backend, expected):
    assert context(launch_module, backend).launch_configurations['hand'] == expected


@pytest.mark.parametrize(('hand', 'left_flag'), [('left', 'true'), ('right', 'false')])
def test_hardware_calls_default_rs485_controller_with_matching_model(launch_module, hand, left_flag):
    from launch.actions import ExecuteProcess, IncludeLaunchDescription
    from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
    ctx = context(launch_module, 'hardware', hand=hand)
    actions = launch_module.start_preview(ctx)
    assert len(actions) == 3
    assert isinstance(actions[0], IncludeLaunchDescription)
    actions[0].launch_description_source.get_launch_description(ctx)
    source = actions[0].launch_description_source.location
    assert source.endswith('/dexhand_bringup/launch/hw_dexhand_controller.launch.py')
    args = dict(actions[0].launch_arguments)
    xacro_args = perform_substitutions(ctx, normalize_to_list_of_substitutions(args['xacro_args']))
    assert f'use_left_hand:={left_flag}' in xacro_args
    assert 'use_mock_dexhand:=false' in xacro_args
    assert 'communicate:=RS485' in xacro_args
    assert 'port_name:=/dev/dexhand' in xacro_args
    for action in actions[1:]:
        assert isinstance(action, ExecuteProcess)
        cmd = [perform_substitutions(ctx, arg) for arg in action.cmd]
        assert cmd[cmd.index('--hand') + 1] == hand
        assert '--record-dir' not in cmd
        assert '--log-dir' not in cmd
        assert '--publish-timing' not in cmd


def test_existing_controller_is_reused_without_starting_another_driver(launch_module):
    from launch.actions import ExecuteProcess
    ctx = context(launch_module, 'hardware', start_hardware='false', hand='right')
    actions = launch_module.start_preview(ctx)
    assert len(actions) == 2
    assert all(isinstance(action, ExecuteProcess) for action in actions)


def test_hardware_logging_is_opt_in_and_wires_both_processes(launch_module, tmp_path):
    from launch.utilities import perform_substitutions
    ctx = context(launch_module, 'hardware', start_hardware='false', hardware_log='true',
                  hardware_log_dir=str(tmp_path))
    relay, tracker = launch_module.start_preview(ctx)
    relay_cmd = [perform_substitutions(ctx, arg) for arg in relay.cmd]
    tracker_cmd = [perform_substitutions(ctx, arg) for arg in tracker.cmd]
    assert relay_cmd[relay_cmd.index('--log-dir') + 1] == str(tmp_path)
    assert '--publish-timing' in tracker_cmd
    assert '--record-dir' not in tracker_cmd


def test_hardware_log_rejects_simulation_backend(launch_module):
    with pytest.raises(ValueError, match='hardware_log requires'):
        launch_module.start_preview(context(launch_module, 'mujoco', hardware_log='true'))
