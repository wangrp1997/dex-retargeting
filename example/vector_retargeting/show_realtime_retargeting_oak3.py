import sys
import os

# 在导入任何库之前禁用 SteamVR/OpenVR
# 修改 LD_LIBRARY_PATH，移除 SteamVR 相关路径
if 'LD_LIBRARY_PATH' in os.environ:
    ld_paths = os.environ['LD_LIBRARY_PATH'].split(':')
    # 过滤掉所有包含 steam 或 SteamVR 的路径
    filtered_paths = [p for p in ld_paths if 'steam' not in p.lower() and 'steamvr' not in p.lower()]
    os.environ['LD_LIBRARY_PATH'] = ':'.join(filtered_paths)

# 设置环境变量禁用 OpenVR/SteamVR
# VR_OVERRIDE 设置为不存在的路径，阻止 SAPIEN 初始化 VR
os.environ['VR_OVERRIDE'] = '/dev/null'
os.environ['OPENVR_INIT'] = '0'
os.environ['STEAMVR_INIT'] = '0'
os.environ['DISABLE_STEAMVR'] = '1'
os.environ['OPENVR_DISABLE'] = '1'
# 禁用 SAPIEN 的 VR 支持
os.environ['SAPIEN_DISABLE_VR'] = '1'

# 重定向 stderr 来隐藏 SteamVR 的日志（可选，如果需要看到其他错误可以注释掉）
# import contextlib
# stderr_fd = os.dup(2)  # 保存原始 stderr
# devnull = os.open(os.devnull, os.O_WRONLY)
# os.dup2(devnull, 2)  # 重定向 stderr 到 /dev/null

# 调整 Python 路径，优先使用 conda 环境中的 pinocchio 而不是 ROS 的版本
# 将 conda 环境的路径移到前面，ROS 路径移到后面
conda_paths = [p for p in sys.path if 'miniconda3' in p or 'conda' in p]
ros_paths = [p for p in sys.path if 'ros' in p and 'site-packages' in p]
other_paths = [p for p in sys.path if p not in conda_paths and p not in ros_paths]
sys.path = conda_paths + other_paths + ros_paths

import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector
from oak import OakCamera


def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    # 添加手部状态跟踪变量
    last_joint_pos = None
    last_keypoint_2d = None
    hand_lost_frames = 0
    HAND_LOST_THRESHOLD = 10  # 连续多少帧检测不到手部才重新检测
    DETECTION_INTERVAL = 5    # 正常检测的间隔帧数
    frame_count = 0

    # 设置使用 GPU 渲染
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)

    # Setup
    scene = sapien.Scene()
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
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # Camera
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.30, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

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
    elif "botyard" in robot_name:
        loader.scale = 0.9

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
    elif "botyard" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2], [-0.707, 0, 0, 0.707]))  # 添加四元数旋转

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    # 创建关键点球体可视化
    print('创建关键点球体可视化...')
    keypoint_spheres = []
    keypoint_radius = 0.008  # 减小到8mm半径
    
    # 定义不同关键点的颜色
    keypoint_colors = {
        'wrist': [1.0, 0.0, 0.0, 1.0],      # 红色 - 手腕
        'thumb': [0.0, 1.0, 0.0, 1.0],      # 绿色 - 拇指
        'index': [0.0, 0.0, 1.0, 1.0],      # 蓝色 - 食指
        'middle': [1.0, 1.0, 0.0, 1.0],     # 黄色 - 中指
        'ring': [1.0, 0.0, 1.0, 1.0],       # 紫色 - 无名指
        'pinky': [0.0, 1.0, 1.0, 1.0],      # 青色 - 小指
    }
    
    # 获取目标link名称
    target_link_names = []
    
    # 从优化器中获取目标link名称
    if hasattr(retargeting.optimizer, 'computed_link_names'):
        # 使用computed_link_names，它包含了所有需要的link
        target_link_names = retargeting.optimizer.computed_link_names
        print(f'从computed_link_names获取目标link: {target_link_names}')
    else:
        # 备用方案：从配置中直接读取
        print('无法从优化器获取link名称，尝试从配置读取...')
        if hasattr(retargeting.optimizer, 'wrist_link_name'):
            target_link_names.append(retargeting.optimizer.wrist_link_name)
        if hasattr(retargeting.optimizer, 'finger_tip_link_names'):
            target_link_names.extend(retargeting.optimizer.finger_tip_link_names)
    
    print(f'最终目标link: {target_link_names}')
    
    # 为每个目标link创建一个球体
    for i, link_name in enumerate(target_link_names):
        # 根据link名称选择颜色
        link_name_lower = link_name.lower()
        if 'thbase' in link_name_lower or 'pmbase' in link_name_lower or 'wrist' in link_name_lower or 'base' in link_name_lower:
            color = keypoint_colors['wrist']  # 红色 - 手腕/基础
        elif 'thtip' in link_name_lower or 'thumb' in link_name_lower:
            color = keypoint_colors['thumb']  # 绿色 - 拇指
        elif 'fftip' in link_name_lower or 'index' in link_name_lower:
            color = keypoint_colors['index']  # 蓝色 - 食指
        elif 'mftip' in link_name_lower or 'middle' in link_name_lower:
            color = keypoint_colors['middle']  # 黄色 - 中指
        elif 'rftip' in link_name_lower or 'ring' in link_name_lower:
            color = keypoint_colors['ring']  # 紫色 - 无名指
        elif 'lftip' in link_name_lower or 'pinky' in link_name_lower or 'little' in link_name_lower:
            color = keypoint_colors['pinky']  # 青色 - 小指
        else:
            # 对于其他link，使用循环颜色
            color_index = i % len(keypoint_colors)
            color_names = list(keypoint_colors.keys())
            color = keypoint_colors[color_names[color_index]]
        
        keypoint_material = sapien.render.RenderMaterial()
        keypoint_material.base_color = color
        keypoint_material.metallic = 0.0
        keypoint_material.roughness = 0.3
        keypoint_material.specular = 0.8
        
        # 使用正确的SAPIEN API创建球体
        builder = scene.create_actor_builder()
        builder.add_sphere_visual(
            radius=keypoint_radius,
            material=keypoint_material
        )
        sphere = builder.build_static(name=f'target_link_{i}')
        # 初始位置设为相机前方，避免被地面遮挡
        sphere.set_pose(sapien.Pose([0.3, 0, 0.1]))
        keypoint_spheres.append(sphere)
    
    print(f'创建了 {len(keypoint_spheres)} 个目标link球体')

    while True:
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            logger.error(
                "Fail to fetch image from camera in 5 secs. Please check your OAK-D camera device."
            )
            return

        frame_count += 1
        need_detection = False

        # 判断是否需要检测手部
        if last_joint_pos is None:
            # 首次运行，需要检测
            need_detection = True
        elif hand_lost_frames >= HAND_LOST_THRESHOLD:
            # 手部丢失超过阈值，需要重新检测
            need_detection = True
        elif frame_count % DETECTION_INTERVAL == 0:
            # 定期检测，确保跟踪准确
            need_detection = True

        if need_detection:
            # 进行手部检测
            _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
            if joint_pos is not None:
                # 检测到手部，更新状态
                last_joint_pos = joint_pos
                last_keypoint_2d = keypoint_2d
                hand_lost_frames = 0
            else:
                # 未检测到手部
                hand_lost_frames += 1
        else:
            # 使用上一帧的检测结果
            joint_pos = last_joint_pos
            keypoint_2d = last_keypoint_2d

        # 绘制骨架
        if keypoint_2d is not None:
            bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        
        # 显示状态信息
        status_text = f"Hand {'Lost' if hand_lost_frames > 0 else 'Tracked'} ({hand_lost_frames} frames)"
        cv2.putText(bgr, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("realtime_retargeting_demo", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if joint_pos is not None:
            # 更新目标link球体位置
            print('检测到手部，更新目标link位置')
            
            # 获取目标link名称
            target_link_names = []
            if hasattr(retargeting.optimizer, 'computed_link_names'):
                target_link_names = retargeting.optimizer.computed_link_names
            else:
                if hasattr(retargeting.optimizer, 'wrist_link_name'):
                    target_link_names.append(retargeting.optimizer.wrist_link_name)
                if hasattr(retargeting.optimizer, 'finger_tip_link_names'):
                    target_link_names.extend(retargeting.optimizer.finger_tip_link_names)
            
            # 获取机器人所有link
            robot_links = robot.get_links()
            link_dict = {link.get_name(): link for link in robot_links}
            
            for i, sphere in enumerate(keypoint_spheres):
                if i < len(target_link_names):
                    link_name = target_link_names[i]
                    if link_name in link_dict:
                        # 获取目标link的世界坐标位置
                        link = link_dict[link_name]
                        link_pose = link.get_pose()
                        link_position = link_pose.p  # 获取位置
                        
                        # 直接使用link的世界坐标位置
                        sphere.set_pose(sapien.Pose(link_position))
                        print(f'球体 {i} (link: {link_name}) 位置: [{link_position[0]:.3f}, {link_position[1]:.3f}, {link_position[2]:.3f}]')
                    else:
                        print(f'警告: 找不到link {link_name}')
                        sphere.set_pose(sapien.Pose([10, 10, 10]))
                else:
                    # 如果球体数量多于目标link，将球体移到远处
                    sphere.set_pose(sapien.Pose([10, 10, 10]))
            
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            elif retargeting_type == "DEXPILOT":
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            elif retargeting_type == "HYBRID":
                indices_vec = retargeting.optimizer.target_link_human_indices_vec
                origin_indices = indices_vec[0, :]
                task_indices = indices_vec[1, :]
                ref_value = {
                    "position": joint_pos[indices, :],
                    "vector": joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                }
            elif retargeting_type == "POSITION_PINCH":
                # 和 hybrid 一样，需要同时提供位置和向量信息
                indices_vec = retargeting.optimizer.target_link_human_indices_vec
                origin_indices = indices_vec[0, :]
                task_indices = indices_vec[1, :]
                ref_value = {
                    "target_pos": joint_pos[indices, :],  # 注意这里用 target_pos 而不是 position
                    "target_vec": joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                }
            else:
                raise ValueError(f"Unsupported retargeting type: {retargeting_type}")
            
            fixed_qpos = np.zeros(3)  # 两个关节都设为 0
            qpos = retargeting.retarget(ref_value, fixed_qpos=fixed_qpos)
            robot.set_qpos(qpos[retargeting_to_sapien])

        # 减少渲染次数
        viewer.render()
        
        # 添加调试信息
        if frame_count % 30 == 0:  # 每30帧打印一次调试信息
            print(f'=== 调试信息 (帧 {frame_count}) ===')
            print(f'相机位置: {cam.get_local_pose().p}')
            print(f'目标link球体数量: {len(keypoint_spheres)}')
            
            # 获取目标link名称
            target_link_names = []
            if hasattr(retargeting.optimizer, 'computed_link_names'):
                target_link_names = retargeting.optimizer.computed_link_names
            else:
                if hasattr(retargeting.optimizer, 'wrist_link_name'):
                    target_link_names.append(retargeting.optimizer.wrist_link_name)
                if hasattr(retargeting.optimizer, 'finger_tip_link_names'):
                    target_link_names.extend(retargeting.optimizer.finger_tip_link_names)
            
            print(f'目标link: {target_link_names}')
            if len(target_link_names) > 0:
                robot_links = robot.get_links()
                link_dict = {link.get_name(): link for link in robot_links}
                if target_link_names[0] in link_dict:
                    first_link = link_dict[target_link_names[0]]
                    print(f'第一个目标link: {first_link.get_name()}, 位置: {first_link.get_pose().p}')
            print('=======================')


def produce_frame(queue: multiprocessing.Queue, resolution: str = "full"):
    try:
        cap = OakCamera(
            input_src="rgb",
            resolution=resolution,
            internal_fps=30,
            xyz=False,
            crop=False,
            internal_frame_height=640
        )
        
        while True:
            ret, frame, _ = cap.read()
            if not ret:
                print("读取帧失败")
                time.sleep(0.1)
                continue
                
            # 确保帧不为空且大小正确
            if frame is None or frame.size == 0:
                print("帧为空")
                continue
                
            # 直接放入队列，不等待
            queue.put(frame, block=False)
            
            # 如果队列满了，清空一下
            if queue.full():
                try:
                    queue.get_nowait()
                except Empty:
                    pass
                    
            time.sleep(1 / 30.0)
            
    except Exception as e:
        print(f"相机错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'cap' in locals():
            cap.release()


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    resolution: str = "full",
):
    """
    使用 OAK-D 相机进行实时手势重定向。

    Args:
        robot_name: 机器人名称，必须是默认支持的机器人之一。
        retargeting_type: 重定向类型，每种类型对应不同的重定向算法。
        hand_type: 指定跟踪哪只手，左手或右手。
        resolution: 相机分辨率，可选 "full" (1280x800) 或 "ultra" (3840x2160)
    """
    # 设置使用 GPU 渲染
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    # 获取配置路径和机器人目录
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    print(f"加载配置文件: {config_path}")
    config = RetargetingConfig.load_from_file(config_path)

    # 初始化重定向
    print("初始化重定向...")
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    
    # 初始化手势检测器
    print("初始化手势检测器...")
    hand_type_str = "Right" if "right" in str(config_path).lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type_str, selfie=False)

    # 初始化 SAPIEN 场景
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # 添加光照
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # 添加相机
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.30, 0, 0.0], [0, 0, 0, -1]))
    
    # 创建查看器
    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # 加载机器人
    print(f"加载机器人 URDF: {config.urdf_path}")
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True
    
    # 设置机器人缩放
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
    elif "botyard" in robot_name:
        loader.scale = 0.9

    filepath = str(filepath)
    robot = loader.load(filepath)

    # 设置机器人位置
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
    elif "botyard" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2], [-0.707, 0, 0, 0.707]))

    # 设置关节映射
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    # 创建关键点球体可视化
    print('创建关键点球体可视化...')
    keypoint_spheres = []
    keypoint_radius = 0.008  # 减小到8mm半径
    
    # 定义不同关键点的颜色
    keypoint_colors = {
        'wrist': [1.0, 0.0, 0.0, 1.0],      # 红色 - 手腕
        'thumb': [0.0, 1.0, 0.0, 1.0],      # 绿色 - 拇指
        'index': [0.0, 0.0, 1.0, 1.0],      # 蓝色 - 食指
        'middle': [1.0, 1.0, 0.0, 1.0],     # 黄色 - 中指
        'ring': [1.0, 0.0, 1.0, 1.0],       # 紫色 - 无名指
        'pinky': [0.0, 1.0, 1.0, 1.0],      # 青色 - 小指
    }
    
    # 获取目标link名称
    target_link_names = []
    
    # 从优化器中获取目标link名称
    if hasattr(retargeting.optimizer, 'computed_link_names'):
        # 使用computed_link_names，它包含了所有需要的link
        target_link_names = retargeting.optimizer.computed_link_names
        print(f'从computed_link_names获取目标link: {target_link_names}')
    else:
        # 备用方案：从配置中直接读取
        print('无法从优化器获取link名称，尝试从配置读取...')
        if hasattr(retargeting.optimizer, 'wrist_link_name'):
            target_link_names.append(retargeting.optimizer.wrist_link_name)
        if hasattr(retargeting.optimizer, 'finger_tip_link_names'):
            target_link_names.extend(retargeting.optimizer.finger_tip_link_names)
    
    print(f'最终目标link: {target_link_names}')
    
    # 为每个目标link创建一个球体
    for i, link_name in enumerate(target_link_names):
        # 根据link名称选择颜色
        link_name_lower = link_name.lower()
        if 'thbase' in link_name_lower or 'pmbase' in link_name_lower or 'wrist' in link_name_lower or 'base' in link_name_lower:
            color = keypoint_colors['wrist']  # 红色 - 手腕/基础
        elif 'thtip' in link_name_lower or 'thumb' in link_name_lower:
            color = keypoint_colors['thumb']  # 绿色 - 拇指
        elif 'fftip' in link_name_lower or 'index' in link_name_lower:
            color = keypoint_colors['index']  # 蓝色 - 食指
        elif 'mftip' in link_name_lower or 'middle' in link_name_lower:
            color = keypoint_colors['middle']  # 黄色 - 中指
        elif 'rftip' in link_name_lower or 'ring' in link_name_lower:
            color = keypoint_colors['ring']  # 紫色 - 无名指
        elif 'lftip' in link_name_lower or 'pinky' in link_name_lower or 'little' in link_name_lower:
            color = keypoint_colors['pinky']  # 青色 - 小指
        else:
            # 对于其他link，使用循环颜色
            color_index = i % len(keypoint_colors)
            color_names = list(keypoint_colors.keys())
            color = keypoint_colors[color_names[color_index]]
        
        keypoint_material = sapien.render.RenderMaterial()
        keypoint_material.base_color = color
        keypoint_material.metallic = 0.0
        keypoint_material.roughness = 0.3
        keypoint_material.specular = 0.8
        
        # 使用正确的SAPIEN API创建球体
        builder = scene.create_actor_builder()
        builder.add_sphere_visual(
            radius=keypoint_radius,
            material=keypoint_material
        )
        sphere = builder.build_static(name=f'target_link_{i}')
        # 初始位置设为相机前方，避免被地面遮挡
        sphere.set_pose(sapien.Pose([0.3, 0, 0.1]))
        keypoint_spheres.append(sphere)
    
    print(f'创建了 {len(keypoint_spheres)} 个目标link球体')

    try:
        # 初始化 OAK-D 相机
        print("正在初始化相机...")
        cap = OakCamera(
            input_src="rgb",
            resolution=resolution,
            internal_fps=30,
            xyz=False,
            crop=False,
            internal_frame_height=640
        )
        print("相机初始化完成")
        
        # 帧率计算相关变量
        fps_counter = 0
        fps_start_time = time.time()
        fps = 0
        
        while True:
            # 读取相机帧
            ret, frame, _ = cap.read()
            if not ret:
                print("读取帧失败")
                continue
            
            # 转换颜色空间
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测手势
            _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
            
            # 计算帧率
            fps_counter += 1
            if fps_counter >= 30:  # 每30帧更新一次帧率
                current_time = time.time()
                fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
            
            # 在图像上绘制骨架
            frame = detector.draw_skeleton_on_image(frame, keypoint_2d, style="default")
            
            # 在图像上显示帧率
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示相机画面
            cv2.imshow("Hand Detection", frame)
            
            # 重定向手势到机器人
            if joint_pos is not None:
                # 更新目标link球体位置
                print('检测到手部，更新目标link位置')
                
                # 获取目标link名称
                target_link_names = []
                if hasattr(retargeting.optimizer, 'computed_link_names'):
                    target_link_names = retargeting.optimizer.computed_link_names
                else:
                    if hasattr(retargeting.optimizer, 'wrist_link_name'):
                        target_link_names.append(retargeting.optimizer.wrist_link_name)
                    if hasattr(retargeting.optimizer, 'finger_tip_link_names'):
                        target_link_names.extend(retargeting.optimizer.finger_tip_link_names)
                
                # 获取机器人所有link
                robot_links = robot.get_links()
                link_dict = {link.get_name(): link for link in robot_links}
                
                for i, sphere in enumerate(keypoint_spheres):
                    if i < len(target_link_names):
                        link_name = target_link_names[i]
                        if link_name in link_dict:
                            # 获取目标link的世界坐标位置
                            link = link_dict[link_name]
                            link_pose = link.get_pose()
                            link_position = link_pose.p  # 获取位置
                            
                            # 直接使用link的世界坐标位置
                            sphere.set_pose(sapien.Pose(link_position))
                            print(f'球体 {i} (link: {link_name}) 位置: [{link_position[0]:.3f}, {link_position[1]:.3f}, {link_position[2]:.3f}]')
                        else:
                            print(f'警告: 找不到link {link_name}')
                            sphere.set_pose(sapien.Pose([10, 10, 10]))
                    else:
                        # 如果球体数量多于目标link，将球体移到远处
                        sphere.set_pose(sapien.Pose([10, 10, 10]))
                
                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices
                if retargeting_type == "POSITION":
                    indices = indices
                    ref_value = joint_pos[indices, :]
                elif retargeting_type == "DEXPILOT" or retargeting_type == "VECTOR":
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                elif retargeting_type == "HYBRID":
                    indices_vec = retargeting.optimizer.target_link_human_indices_vec
                    origin_indices = indices_vec[0, :]
                    task_indices = indices_vec[1, :]
                    ref_value = {
                        "position": joint_pos[indices, :],
                        "vector": joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    }
                elif retargeting_type == "POSITION_PINCH":
                    indices_vec = retargeting.optimizer.target_link_human_indices_vec
                    origin_indices = indices_vec[0, :]
                    task_indices = indices_vec[1, :]
                    ref_value = {
                        "target_pos": joint_pos[indices, :],
                        "target_vec": joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    }
                else:
                    raise ValueError(f"Unsupported retargeting type: {retargeting_type}")
                
                fixed_qpos = np.zeros(2)
                qpos = retargeting.retarget(ref_value, fixed_qpos=fixed_qpos)
                robot.set_qpos(qpos[retargeting_to_sapien])
            else:
                print("未检测到手")
            
            # 渲染 SAPIEN 场景
            viewer.render()
            
            # 添加调试信息
            if fps_counter % 30 == 0:  # 每30帧打印一次调试信息
                print(f'=== 调试信息 (帧 {fps_counter}) ===')
                print(f'相机位置: {cam.get_local_pose().p}')
                print(f'目标link球体数量: {len(keypoint_spheres)}')
                
                # 获取目标link名称
                target_link_names = []
                if hasattr(retargeting.optimizer, 'computed_link_names'):
                    target_link_names = retargeting.optimizer.computed_link_names
                else:
                    if hasattr(retargeting.optimizer, 'wrist_link_name'):
                        target_link_names.append(retargeting.optimizer.wrist_link_name)
                    if hasattr(retargeting.optimizer, 'finger_tip_link_names'):
                        target_link_names.extend(retargeting.optimizer.finger_tip_link_names)
                
                print(f'目标link: {target_link_names}')
                if len(target_link_names) > 0:
                    robot_links = robot.get_links()
                    link_dict = {link.get_name(): link for link in robot_links}
                    if target_link_names[0] in link_dict:
                        first_link = link_dict[target_link_names[0]]
                        print(f'第一个目标link: {first_link.get_name()}, 位置: {first_link.get_pose().p}')
                print('=======================')
            
            # 检查退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(1 / 30.0)
            
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tyro.cli(main) 