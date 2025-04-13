from pathlib import Path
from typing import Optional, List, Union, Dict

import cv2
import numpy as np
import sapien
import tqdm
import tyro
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer
# import isaacgym
# import isaacgym.gymapi as gymapi
import pybullet as p
import pybullet_data

from dex_retargeting.retargeting_config import RetargetingConfig

# Convert webp
# ffmpeg -i teaser.mp4 -vcodec libwebp -lossless 1 -loop 0 -preset default  -an -vsync 0 teaser.webp


def render_by_isaacgym(
    meta_data: Dict,
    data: List[Union[List[float], np.ndarray]],
    output_video_path: Optional[str] = None,
    headless: Optional[bool] = False,
):
    # 初始化 gym
    gym = gymapi.acquire_gym()

    # 创建模拟器
    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt = 1.0 / 120.0
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.use_gpu = False
    sim_params.up_axis = gymapi.UP_AXIS_Z

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    if sim is None:
        print("*** Failed to create sim")
        return

    # 创建viewer
    if not headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise ValueError("*** Failed to create viewer")
    else:
        viewer = None
    record_video = output_video_path is not None

    # 添加地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.distance = 0.5
    gym.add_ground(sim, plane_params)

    # 设置环境网格
    spacing = 2
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, 0.0, spacing)

    # 加载URDF
    config_path = meta_data["config_path"]
    config = RetargetingConfig.load_from_file(config_path)
    asset_path = Path(config.urdf_path)
    asset_root = str(asset_path.parent.parent)
    asset_name = asset_path.stem
    asset_file = f"{asset_path.parent.name}/{asset_path.name}"

    # 加载资产
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = False
    asset_options.convex_decomposition_from_submeshes = True
    asset_options.disable_gravity = True
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    num_dof = gym.get_asset_dof_count(asset)

    # 创建环境
    env = gym.create_env(sim, env_lower, env_upper, 2)
    initial_pose = gymapi.Transform()
    actor = gym.create_actor(env, asset, initial_pose, asset_name, 0, 0)

    # 设置相机
    if record_video:
        cam_prop = gymapi.CameraProperties()
        cam_prop.width = 600
        cam_prop.height = 600
        cam_prop.enable_tensors = False
        cam_prop.horizontal_fov = np.rad2deg(1)
        cam = gym.create_camera_sensor(env, cam_prop)
        cam_quat = gymapi.Quat(0, 0, 0, -1)
        cam_pos = gymapi.Vec3(0.50, 0, 0.0)
        cam_pose = gymapi.Transform(cam_pos, cam_quat)
        gym.set_camera_transform(cam, env, cam_pose)

    # 视频记录器
    if record_video:
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (600, 600),
        )

    # 渲染循环
    for qpos in tqdm.tqdm(data):
        # 设置关节位置
        dof_state = gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)
        dof_state["pos"] = qpos
        gym.set_actor_dof_states(env, actor, dof_state, gymapi.STATE_POS)
        
        # 更新物理
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)

        # 更新viewer
        if not headless:
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        # 更新视频
        if record_video:
            gym.render_all_camera_sensors(sim)
            rgb = gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR).reshape([600, -1, 4])[..., :3]
            writer.write(rgb[..., ::-1])

    if record_video:
        writer.release()

    if not headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


def render_by_pybullet(
    meta_data: Dict,
    data: List[Union[List[float], np.ndarray]],
    output_video_path: Optional[str] = None,
    headless: Optional[bool] = False,
):
    # 初始化PyBullet
    if headless:
        physicsClient = p.connect(p.DIRECT)
    else:
        physicsClient = p.connect(p.GUI)
    
    # 基本设置
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # 加载地面
    planeId = p.loadURDF("plane.urdf")
    
    # 加载URDF
    config_path = meta_data["config_path"]
    config = RetargetingConfig.load_from_file(config_path)
    urdf_path = config.urdf_path
    print(f"Loading URDF from: {urdf_path}")
    
    # 设置URDF加载选项
    flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
    # 调整手的初始位置，使其在地面上方
    robotId = p.loadURDF(urdf_path, [0, 0, 0.1], useFixedBase=True, flags=flags)
    
    # 获取关节信息
    num_joints = p.getNumJoints(robotId)
    joint_indices = [i for i in range(num_joints) if p.getJointInfo(robotId, i)[2] != p.JOINT_FIXED]
    
    # 设置相机 - 调整到正视图
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8,  # 增加距离，让整个手可见
        cameraYaw=90,  # 保持正面视角
        cameraPitch=0,  # 调整为水平视角
        cameraTargetPosition=[0, 0, 0.2]  # 对准手掌中心
    )
    
    # 视频记录器
    if output_video_path:
        print(f"Creating video at: {output_video_path}")
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        width, height = 600, 600
        
        # 创建临时目录存储帧
        temp_dir = Path(output_video_path).parent / "temp_frames"
        temp_dir.mkdir(exist_ok=True)
        
        # 渲染循环
        frame_count = 0
        for qpos in tqdm.tqdm(data):
            # 设置关节位置
            for i, joint_idx in enumerate(joint_indices):
                if i < len(qpos):  # 确保不超出输入数据的范围
                    p.resetJointState(robotId, joint_idx, qpos[i])
                else:
                    print(f"Warning: Joint index {i} exceeds input data length {len(qpos)}")
            
            # 更新物理
            p.stepSimulation()
            
            # 获取相机图像 - 使用正视图
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.2],
                distance=0.8,
                yaw=90,
                pitch=0,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=100.0
            )
            
            # 获取图像并转换为numpy数组
            width, height, rgb, depth, seg = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # 将RGB数据转换为numpy数组
            rgb_array = np.array(rgb, dtype=np.uint8)
            rgb_array = rgb_array.reshape((height, width, 4))  # RGBA格式
            
            # 保存帧
            rgb_array = rgb_array[..., :3]  # 只取RGB通道
            rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            frame_path = temp_dir / f"frame_{frame_count:04d}.png"
            cv2.imwrite(str(frame_path), rgb_array)
            frame_count += 1
        
        # 使用ffmpeg合并帧
        print("Combining frames with ffmpeg...")
        import subprocess
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", "30",
            "-i", str(temp_dir / "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            str(output_video_path)
        ]
        subprocess.run(cmd, check=True)
        
        # 清理临时文件
        for frame_file in temp_dir.glob("*.png"):
            frame_file.unlink()
        temp_dir.rmdir()
        print("Video writing completed")
    
    p.disconnect()


def render_by_sapien(
    meta_data: Dict,
    data: List[Union[List[float], np.ndarray]],
    output_video_path: Optional[str] = None,
    headless: Optional[bool] = False,
):
    # 使用简单的渲染设置
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")
    
    # 创建场景
    scene = sapien.Scene()

    # Config is loaded only to find the urdf path and robot name
    config_path = meta_data["config_path"]
    config = RetargetingConfig.load_from_file(config_path)

    # Ground
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )

    # Camera
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    # Viewer
    if not headless:
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame = False
        viewer.control_window.move_speed = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(cam.get_local_pose())
    else:
        viewer = None
    record_video = output_video_path is not None

    # Load robot and set it to a good pose to take picture
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True
    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5

    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)
    robot = loader.load(filepath)

    if "ability" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.13]))
    elif "inspire" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))

    # Video recorder
    if record_video:
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (cam.get_width(), cam.get_height()),
        )

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = meta_data["joint_names"]
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    for qpos in tqdm.tqdm(data):
        robot.set_qpos(np.array(qpos)[retargeting_to_sapien])

        if not headless:
            for _ in range(2):
                viewer.render()
        if record_video:
            scene.update_render()
            cam.take_picture()
            rgb = cam.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            writer.write(rgb[..., ::-1])

    if record_video:
        writer.release()

    scene = None


def main(
    pickle_path: str,
    output_video_path: Optional[str] = None,
    headless: bool = False,
    renderer: str = "pybullet",  # 默认使用PyBullet
):
    """
    Loads the preserved robot pose data and renders it either on screen or as an mp4 video.

    Args:
        pickle_path: Path to the .pickle file, created by `detect_from_video.py`.
        output_video_path: Path where the output video in .mp4 format would be saved.
            By default, it is set to None, implying no video will be saved.
        headless: Set to visualize the rendering on the screen by opening the viewer window.
        renderer: Choose the renderer to use, either "sapien" or "pybullet".
    """
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    pickle_data = np.load(pickle_path, allow_pickle=True)
    meta_data, data = pickle_data["meta_data"], pickle_data["data"]

    if renderer == "pybullet":
        render_by_pybullet(meta_data, data, output_video_path, headless)
    else:
        render_by_sapien(meta_data, data, output_video_path, headless)


if __name__ == "__main__":
    tyro.cli(main)
