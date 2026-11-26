"""Launch GS S1 webcam retargeting with the RViz or MuJoCo backend."""

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
from launch.substitutions import LaunchConfiguration


def start_preview(context):
    example = Path(__file__).resolve().parent
    src = example.parents[1] / "src"
    backend = LaunchConfiguration("backend").perform(context)
    hand = LaunchConfiguration("hand").perform(context)
    share = Path(get_package_share_directory("dexhand_description"))
    video = LaunchConfiguration("video").perform(context)
    input_args = ["--video", video] if video else ["--camera", LaunchConfiguration("camera")]
    input_args.extend(["--hand", hand])
    input_args.extend(["--objective", LaunchConfiguration("objective"),
                       "--smoothing", LaunchConfiguration("smoothing"),
                       "--assets", LaunchConfiguration("mujoco_assets"),
                       "--wuji-root", LaunchConfiguration("wuji_root"),
                       "--tsinghua-root", LaunchConfiguration("tsinghua_root")])
    tracker_env = {"PYTHONPATH": os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")])}
    if LaunchConfiguration("objective").perform(context) == "botyard":
        tracker_env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        input_args.extend(["--botyard-gradient", LaunchConfiguration("botyard_gradient")])
        mano_model = LaunchConfiguration("mano_model").perform(context)
        if mano_model:
            input_args.extend(["--mano-model", mano_model])
    if not IfCondition(LaunchConfiguration("show_camera")).evaluate(context):
        input_args.append("--no-camera-window")
    if backend == "mujoco":
        input_args.append("--no-target-collision-overlay")
        model = ExecuteProcess(
            cmd=[LaunchConfiguration("mujoco_python"), str(example / "gs_s1_mujoco.py"),
                 "--assets", LaunchConfiguration("mujoco_assets"), "--hand", hand,
                 "--mesh-dir", str(share / "meshes/gs_s1")],
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
    return [model, tracker]


def generate_launch_description():
    projects = Path(__file__).resolve().parents[3]
    local_mujoco = Path.home() / "miniconda3/envs/DGBench/bin/python"
    local_tracker = Path.home() / "miniconda3/envs/dex_retargeting/bin/python"
    return LaunchDescription([
        DeclareLaunchArgument("backend", default_value="rviz", choices=["rviz", "mujoco"]),
        DeclareLaunchArgument("hand", default_value="right", choices=["right", "left"]),
        DeclareLaunchArgument("objective", default_value="dexpilot", choices=["dexpilot", "wuji", "vector", "tsinghua", "botyard"]),
        DeclareLaunchArgument("botyard_gradient", default_value="finite_difference", choices=["finite_difference", "analytic"]),
        DeclareLaunchArgument("smoothing", default_value="one_euro", choices=["one_euro", "off"]),
        DeclareLaunchArgument("wuji_root", default_value=str(projects / "wuji-retargeting")),
        DeclareLaunchArgument("tsinghua_root", default_value=str(projects / "tsinghua-retargeting")),
        DeclareLaunchArgument("mano_model", default_value="", description="Optional prepared MANO NPZ for the selected hand"),
        DeclareLaunchArgument("camera", default_value="0"),
        DeclareLaunchArgument("video", default_value=""),
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
