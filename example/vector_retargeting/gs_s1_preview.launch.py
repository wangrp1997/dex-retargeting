"""Launch GS S1 webcam retargeting with RViz, MuJoCo or the default hardware controller."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EqualsSubstitution, IfElseSubstitution, LaunchConfiguration


def start_preview(context):
    example = Path(__file__).resolve().parent
    src = example.parents[1] / "src"
    backend = LaunchConfiguration("backend").perform(context)
    hand = LaunchConfiguration("hand").perform(context)
    botyard = LaunchConfiguration("objective").perform(context) == "botyard"
    share = Path(get_package_share_directory("dexhand_description"))
    video = LaunchConfiguration("video").perform(context)
    input_args = ["--video", video] if video else ["--camera", LaunchConfiguration("camera")]
    record_dir = LaunchConfiguration("record_dir").perform(context)
    hardware_log = IfCondition(LaunchConfiguration('hardware_log')).evaluate(context)
    if hardware_log and backend != 'hardware':
        raise ValueError('hardware_log requires backend:=hardware')
    if record_dir:
        if backend != "mujoco" or not botyard:
            raise ValueError("Recording requires backend:=mujoco and objective:=botyard")
        input_args.extend(["--record-dir", record_dir])
    input_args.extend(["--hand", hand])
    input_args.extend(["--objective", LaunchConfiguration("objective"),
                       "--smoothing", LaunchConfiguration("smoothing"),
                       "--assets", LaunchConfiguration("mujoco_assets"),
                       "--wuji-root", LaunchConfiguration("wuji_root"),
                       "--tsinghua-root", LaunchConfiguration("tsinghua_root")])
    tracker_env = {"PYTHONPATH": os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")])}
    startup = []
    if botyard:
        tracker_env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        input_args.extend(["--botyard-gradient", LaunchConfiguration("botyard_gradient")])
        input_args.extend(["--botyard-shape", LaunchConfiguration("botyard_shape")])
        input_args.extend(['--live-pipeline', LaunchConfiguration('live_pipeline')])
        mano_model = LaunchConfiguration("mano_model").perform(context)
        if mano_model:
            input_args.extend(["--mano-model", mano_model])
    if not IfCondition(LaunchConfiguration("show_camera")).evaluate(context):
        input_args.append("--no-camera-window")
    if backend == "hardware":
        if not botyard:
            raise ValueError('The hardware backend uses objective:=botyard')
        namespace = LaunchConfiguration('controller_namespace').perform(context).strip('/')
        if IfCondition(LaunchConfiguration('start_hardware')).evaluate(context):
            hardware = Path(get_package_share_directory('dexhand_bringup'))
            port = LaunchConfiguration('hardware_port').perform(context)
            startup.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(hardware / 'launch/hw_dexhand_controller.launch.py')),
                launch_arguments={
                    'namespace': namespace, 'use_rviz': 'False', 'cap_grant': 'shell',
                    'xacro_args': ('sub_version:=2 use_ros2_control:=true use_gazebo:=false use_mock_dexhand:=false '
                                   f'use_left_hand:={str(hand == "left").lower()} '
                                   f'communicate:=RS485 port_name:={port}'),
                }.items(),
            ))
        input_args.append('--no-target-collision-overlay')
        log_args = []
        if hardware_log:
            log_args = ['--log-dir', LaunchConfiguration('hardware_log_dir')]
            input_args.append('--publish-timing')
        model = ExecuteProcess(
            cmd=[LaunchConfiguration('tracker_python'), str(example / 'gs_s1_hardware.py'),
                 '--hand', hand, '--controller-namespace', namespace, *log_args],
            output='screen',
            on_exit=[EmitEvent(event=Shutdown(reason='Hardware relay closed'))],
        )
    elif backend == "mujoco":
        overlay_args = []
        if botyard:
            template = mano_model or str(Path.home() / '.cache/botyard-retargeting/mano' / f'{hand}.npz')
            overlay_args = ["--overlay-template", template]
        input_args.append("--no-target-collision-overlay")
        model = ExecuteProcess(
            cmd=[LaunchConfiguration("mujoco_python"), str(example / "gs_s1_mujoco.py"),
                 "--assets", LaunchConfiguration("mujoco_assets"), "--hand", hand,
                 "--mesh-dir", str(share / "meshes/gs_s1"), *overlay_args, *(["--record-view"] if record_dir else [])],
            output="screen",
            on_exit=[EmitEvent(event=Shutdown(reason="MuJoCo viewer closed"))],
        )
    else:
        model = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / "launch/description.launch.py")),
            launch_arguments={
                "namespace": "gs_s1_preview",
                "xacro_args": (
                    f"dexhand_type:=gs_s1 sub_version:=2 use_left_hand:={str(hand == 'left').lower()} "
                    "use_ros2_control:=false use_gazebo:=false"
                ),
                "use_jsp": "False",
                "use_rviz": LaunchConfiguration("use_rviz"),
            }.items(),
        )
    tracker = ExecuteProcess(
        cmd=[LaunchConfiguration("tracker_python"), str(example / "gs_s1_rviz.py"), *input_args],
        additional_env=tracker_env,
        output="screen",
        on_exit=[EmitEvent(event=Shutdown(reason="Camera or video stream finished"))],
    )
    return [*startup, model, tracker]


def generate_launch_description():
    projects = Path(__file__).resolve().parents[3]
    local_mujoco = Path.home() / "miniconda3/envs/DGBench/bin/python"
    local_tracker = Path.home() / "miniconda3/envs/dex_retargeting/bin/python"
    return LaunchDescription([
        DeclareLaunchArgument("backend", default_value="rviz", choices=["rviz", "mujoco", "hardware"]),
        DeclareLaunchArgument("hand", default_value=IfElseSubstitution(
            EqualsSubstitution(LaunchConfiguration('backend'), 'hardware'),
            if_value='left', else_value='right'), choices=["right", "left"]),
        DeclareLaunchArgument("objective", default_value="dexpilot", choices=["dexpilot", "wuji", "vector", "tsinghua", "botyard"]),
        DeclareLaunchArgument("botyard_gradient", default_value="analytic", choices=["finite_difference", "analytic"]),
        DeclareLaunchArgument("botyard_shape", default_value="v4_2", choices=["v4_2", "sweep", "direction", "baseline", "phalanx", "thumb_span"]),
        DeclareLaunchArgument("smoothing", default_value="one_euro", choices=["one_euro", "off"]),
        DeclareLaunchArgument('live_pipeline', default_value='latest', choices=['latest', 'serial'],
                              description='Botyard webcam scheduling; video and recording stay synchronized'),
        DeclareLaunchArgument("wuji_root", default_value=str(projects / "wuji-retargeting")),
        DeclareLaunchArgument("tsinghua_root", default_value=str(projects / "tsinghua-retargeting")),
        DeclareLaunchArgument("mano_model", default_value="", description="Optional prepared MANO NPZ for the selected hand"),
        DeclareLaunchArgument("camera", default_value="0"),
        DeclareLaunchArgument("video", default_value=""),
        DeclareLaunchArgument("record_dir", default_value="", description="Session directory created by record_gs_s1.zsh"),
        DeclareLaunchArgument('hardware_port', default_value='/dev/dexhand', description='RS485 device for the hardware backend'),
        DeclareLaunchArgument('hardware_log', default_value='false', choices=['true', 'false'],
                              description='Save hardware targets, commands, feedback and timings; no video'),
        DeclareLaunchArgument('hardware_log_dir', default_value=str(Path(__file__).resolve().parents[2] / 'logs/hardware'),
                              description='Parent directory for timestamped hardware diagnostic sessions'),
        DeclareLaunchArgument('controller_namespace', default_value='', description='Namespace of the default dexhand_controller'),
        DeclareLaunchArgument('start_hardware', default_value='true', choices=['true', 'false'],
                              description='Start the ros2_botyard default controller; false reuses an existing controller'),
        DeclareLaunchArgument("tracker_python", default_value=(
            str(local_tracker) if local_tracker.is_file() else "python"
        )),
        DeclareLaunchArgument("mujoco_python", default_value=(
            str(local_mujoco) if local_mujoco.is_file() else LaunchConfiguration("tracker_python")
        )),
        DeclareLaunchArgument("mujoco_assets", default_value=str(projects / "tendon-spin/assets")),
        DeclareLaunchArgument("use_rviz", default_value="True"),
        DeclareLaunchArgument("show_camera", default_value=LaunchConfiguration("use_rviz")),
        OpaqueFunction(function=start_preview),
    ])
