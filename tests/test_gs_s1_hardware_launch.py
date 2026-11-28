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
    result.launch_configurations.update(backend=backend, objective='botyard')
    result.launch_configurations.update(overrides)
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


@pytest.mark.parametrize('mode', ['v4_2', 'inward'])
def test_opposition_selection_reaches_tracker_and_log_metadata(launch_module, mode):
    from launch.utilities import perform_substitutions
    ctx = context(launch_module, 'hardware', start_hardware='false', botyard_opposition=mode)
    relay, tracker = launch_module.start_preview(ctx)
    for action in (relay, tracker):
        cmd = [perform_substitutions(ctx, arg) for arg in action.cmd]
        assert cmd[cmd.index('--botyard-opposition') + 1] == mode
    assert context(launch_module, 'hardware').launch_configurations['botyard_opposition'] == 'v4_2'


@pytest.mark.parametrize('overrides', [dict(objective='dexpilot'), dict(botyard_shape='sweep')])
def test_inward_rejects_incompatible_objective_before_bringup(launch_module, overrides):
    ctx = context(launch_module, 'hardware', botyard_opposition='inward', **overrides)
    with pytest.raises(ValueError, match='requires objective:=botyard and botyard_shape:=v4_2'):
        launch_module.start_preview(ctx)


def test_hardware_log_rejects_simulation_backend(launch_module):
    with pytest.raises(ValueError, match='hardware_log requires'):
        launch_module.start_preview(context(launch_module, 'mujoco', hardware_log='true'))


@pytest.mark.parametrize('overrides', [dict(hardware_log='false'), dict(hardware_log='true', start_hardware='false')])
def test_io_trace_requires_logging_and_new_controller_process(launch_module, overrides):
    with pytest.raises(ValueError, match='hardware_io_trace requires'):
        launch_module.start_preview(context(launch_module, 'hardware', hardware_io_trace='true', **overrides))


def test_io_trace_is_scoped_to_hardware_bringup(launch_module, tmp_path):
    from launch.actions import GroupAction, IncludeLaunchDescription, SetEnvironmentVariable
    from launch.utilities import perform_substitutions
    ctx = context(launch_module, 'hardware', hardware_log='true', hardware_io_trace='true',
                  hardware_log_dir=str(tmp_path))
    group, relay, tracker = launch_module.start_preview(ctx)
    assert isinstance(group, GroupAction)
    entities = list(group.get_sub_entities())
    assert sum(isinstance(entity, SetEnvironmentVariable) for entity in entities) == 3
    assert any(isinstance(entity, IncludeLaunchDescription) for entity in entities)
    relay_cmd = [perform_substitutions(ctx, arg) for arg in relay.cmd]
    directory = Path(relay_cmd[relay_cmd.index('--log-session-dir') + 1])
    assert directory.parent == tmp_path
    assert directory.is_dir()
    assert '--log-session-dir' not in [perform_substitutions(ctx, arg) for arg in tracker.cmd]


def test_delayed_spawner_retains_namespace_after_trace_environment_is_restored(launch_module):
    from launch import LaunchDescription, LaunchDescriptionSource, LaunchService
    from launch.actions import IncludeLaunchDescription, OpaqueFunction, TimerAction
    from launch.substitutions import LaunchConfiguration
    observed = []

    def immediate(ctx):
        observed.append(('immediate', LaunchConfiguration('namespace').perform(ctx),
                         ctx.environment.get('BOTYARD_IO_TRACE_FILE')))

    def delayed(ctx):
        # Reproduce the default bringup's TimerAction, after the surrounding
        # environment has been popped. This used to lose namespace entirely.
        observed.append(('delayed', LaunchConfiguration('namespace').perform(ctx),
                         ctx.environment.get('BOTYARD_IO_TRACE_FILE')))

    include = IncludeLaunchDescription(LaunchDescriptionSource(LaunchDescription([
        OpaqueFunction(function=immediate),
        TimerAction(period=.01, actions=[OpaqueFunction(function=delayed)]),
    ])), launch_arguments={'namespace': 'botyard_delayed_test'}.items())
    service = LaunchService()
    previous_trace_file = service.context.environment.get('BOTYARD_IO_TRACE_FILE')
    service.include_launch_description(LaunchDescription([
        launch_module.traced_bringup(include, '/tmp/unused-preload.so', '/tmp/test-io.jsonl', '/tmp/unused-sdk.so')]))
    assert service.run() == 0
    assert observed == [('immediate', 'botyard_delayed_test', '/tmp/test-io.jsonl'),
                        ('delayed', 'botyard_delayed_test', previous_trace_file)]
