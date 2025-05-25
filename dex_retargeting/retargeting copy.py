#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
# 保留原有的导入
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
import os
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from dex_retargeting.src.dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.src.dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.src.single_hand_detector import SingleHandDetector


def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str, publisher, headless: bool = False):
    # 移除 rclpy.init() 调用
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    # 定义目标关节顺序
    target_joint_order = [
        'FAJ1', 'FAJ3', 'FFJ1', 'FFJ2', 'FFJ3', 'FFJ4',
        'LFJ1', 'LFJ2', 'LFJ3', 'LFJ4', 'LFJ5',
        'MFJ1', 'MFJ2', 'MFJ3', 'MFJ4',
        'RFJ1', 'RFJ2', 'RFJ3', 'RFJ4',
        'THJ1', 'THJ2', 'THJ3', 'THJ4'
    ]

    # 添加手部状态跟踪变量
    last_joint_pos = None
    last_keypoint_2d = None
    hand_lost_frames = 0
    HAND_LOST_THRESHOLD = 10  # 连续多少帧检测不到手部才重新检测
    DETECTION_INTERVAL = 5    # 正常检测的间隔帧数
    frame_count = 0

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

    # 根据 headless 模式决定是否创建 viewer
    viewer = None if headless else Viewer()
    if not headless:
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

    if "glb" not in robot_name:
        # filepath = str(filepath).replace(".urdf", "_glb.urdf")
        filepath = str(filepath)
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
                "Fail to fetch image from camera in 5 secs. Please check your web camera device."
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
                # logger.warning(f"{hand_type} hand is not detected. Lost frames: {hand_lost_frames}")
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
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = retargeting.retarget(ref_value)
            robot.set_qpos(qpos[retargeting_to_sapien])
            
            # 创建关节角度映射
            joint_angles = {}
            for i, joint_name in enumerate(sapien_joint_names):
                joint_angles[joint_name] = qpos[retargeting_to_sapien][i]
            
            # 按照目标顺序构建消息
            msg = Float64MultiArray()
            msg.layout.dim.append(MultiArrayDimension())
            msg.layout.dim.append(MultiArrayDimension())
            msg.layout.dim[0].label = "joint_list"
            msg.layout.dim[0].size = len(target_joint_order)
            msg.layout.dim[0].stride = len(target_joint_order)
            msg.layout.dim[1].label = "joint"
            msg.layout.dim[1].size = 1
            msg.layout.dim[1].stride = 1
            
            # 按照目标顺序填充数据
            msg.data = [joint_angles.get(joint_name, 0.0) for joint_name in target_joint_order]
            
            # 发布消息
            publisher.publish(msg)
            
            # 打印关节状态
            joint_status = "\n".join([f"{joint_name}: {msg.data[i]:.4f}" for i, joint_name in enumerate(target_joint_order)])
            logger.info(f"\n发布到话题的关节顺序:\n{'-' * 50}\n{joint_status}\n{'-' * 50}")

        # 根据 headless 模式决定是否渲染
        if not headless:
            viewer.render()
        else:
            # 在 headless 模式下，我们仍然需要更新场景
            scene.update_render()


def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)

    while cap.isOpened():
        success, image = cap.read()
        time.sleep(1 / 30.0)
        if not success:
            continue
        queue.put(image)


class RetargetingNode(Node):
    def __init__(self):
        super().__init__('retargeting_node')
        self.get_logger().info('Retargeting node started')
        
        # 创建发布者
        self.publisher = self.create_publisher(Float64MultiArray, '/dexhand_position_controller/commands', 10)
        
        # 保持原有的参数处理方式
        self.declare_parameters(
            namespace='',
            parameters=[
                ('robot_name', 'botyard'),
                ('retargeting_type', 'POSITION'),
                ('hand_type', 'Left'),
                ('camera_path', 'none'),  # 使用字符串 'none' 代替 None
                ('headless', False)  # 添加 headless 模式参数
            ]
        )
        
        # 启动原有的处理流程
        self.start_retargeting_process()

    def start_retargeting_process(self):
        # 使用原有的参数获取方式
        robot_name = RobotName[self.get_parameter('robot_name').value.lower()]
        retargeting_type = RetargetingType[self.get_parameter('retargeting_type').value.lower()]
        hand_type = HandType[self.get_parameter('hand_type').value.lower()]
        camera_path = self.get_parameter('camera_path').value
        headless = self.get_parameter('headless').value
        if camera_path.lower() == 'none':  # 如果是字符串 'none'，则转换为 None
            camera_path = None

        # 使用原有的配置路径获取方式
        config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
        if config_path is None:
            raise ValueError(f"找不到配置文件：robot_name={robot_name}, retargeting_type={retargeting_type}, hand_type={hand_type}")
        # 直接使用 src 目录下的 URDF 文件
        robot_dir = Path(os.path.expanduser("~/ros2_ws/src/ros2_botyard/dex_retargeting/dex_retargeting/assets/robots/hands"))  # 展开 ~ 符号

        # 创建进程间通信队列
        queue = multiprocessing.Queue(maxsize=1000)
        
        # 启动原有的进程
        producer_process = multiprocessing.Process(
            target=produce_frame, args=(queue, camera_path)
        )
        consumer_process = multiprocessing.Process(
            target=start_retargeting, args=(queue, str(robot_dir), str(config_path), self.publisher, headless)
        )

        producer_process.start()
        consumer_process.start()


def main(args=None):
    rclpy.init(args=args)
    node = RetargetingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
