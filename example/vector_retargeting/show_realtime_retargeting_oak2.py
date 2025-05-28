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
            
            # 在图像上绘制骨架
            frame = detector.draw_skeleton_on_image(frame, keypoint_2d, style="default")
            
            # 显示相机画面
            cv2.imshow("Hand Detection", frame)
            
            # 重定向手势到机器人
            if joint_pos is not None:
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
                    indices_vec = retargeting.optimizer.target_link_human_indices_vec
                    origin_indices = indices_vec[0, :]
                    task_indices = indices_vec[1, :]
                    ref_value = {
                        "target_pos": joint_pos[indices, :],
                        "target_vec": joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    }
                else:
                    raise ValueError(f"Unsupported retargeting type: {retargeting_type}")
                
                fixed_qpos = np.zeros(3)
                qpos = retargeting.retarget(ref_value, fixed_qpos=fixed_qpos)
                robot.set_qpos(qpos[retargeting_to_sapien])
            else:
                print("未检测到手")
            
            # 渲染 SAPIEN 场景
            viewer.render()
            
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