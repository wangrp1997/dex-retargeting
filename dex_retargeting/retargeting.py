#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from loguru import logger
import os
from pathlib import Path
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import time
import sapien
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.src.dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.src.dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.src.single_hand_detector import SingleHandDetector


class RetargetingNode(Node):
    def __init__(self):
        super().__init__('retargeting_node')
        self.get_logger().info('Retargeting node started')
        
        # 创建发布者
        self.publisher = self.create_publisher(Float64MultiArray, '/dexhand_position_controller/commands', 10)
        
        # 声明参数
        self.declare_parameters(
            namespace='',
            parameters=[
                ('robot_name', 'botyard'),
                ('retargeting_type', 'POSITION'),
                ('hand_type', 'Left'),
                ('camera_path', 'none'),
                ('headless', False)  # 添加 headless 参数
            ]
        )
        
        # 获取参数
        robot_name = RobotName[self.get_parameter('robot_name').value.lower()]
        retargeting_type = RetargetingType[self.get_parameter('retargeting_type').value.lower()]
        hand_type = HandType[self.get_parameter('hand_type').value.lower()]
        camera_path = self.get_parameter('camera_path').value
        self.headless = self.get_parameter('headless').value
        if camera_path.lower() == 'none':
            camera_path = None

        # 获取配置文件路径
        config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
        if config_path is None:
            raise ValueError(f"找不到配置文件：robot_name={robot_name}, retargeting_type={retargeting_type}, hand_type={hand_type}")
        
        # 设置机器人目录
        robot_dir = Path(os.path.expanduser("~/ros2_ws/src/ros2_botyard/dex_retargeting/dex_retargeting/assets/robots/hands"))
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        
        # 初始化重定向器
        self.retargeting = RetargetingConfig.load_from_file(config_path).build()
        
        # 初始化手部检测器
        self.hand_type = "Right" if "right" in str(config_path).lower() else "Left"
        self.detector = SingleHandDetector(hand_type=self.hand_type, selfie=False)
        
        # 定义目标关节顺序（按照控制器配置文件的顺序）
        self.target_joint_order = [
            'FAJ1', 'FAJ3',  # 前臂关节
            'FFJ1', 'FFJ2', 'FFJ3', 'FFJ4',  # 食指关节
            'LFJ1', 'LFJ2', 'LFJ3', 'LFJ4', 'LFJ5',  # 小指关节
            'MFJ1', 'MFJ2', 'MFJ3', 'MFJ4',  # 中指关节
            'RFJ1', 'RFJ2', 'RFJ3', 'RFJ4',  # 无名指关节
            'THJ1', 'THJ2', 'THJ3', 'THJ4'  # 拇指关节
        ]
        
        # 获取关节名称映射
        self.retargeting_joint_names = self.retargeting.joint_names
        
        # 初始化 SAPIEN 场景
        if not self.headless:
            self.setup_sapien_scene(config_path)
        
        # 初始化相机
        if camera_path is None:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(camera_path)
            
        # 初始化状态变量
        self.last_joint_pos = None
        self.last_keypoint_2d = None
        self.hand_lost_frames = 0
        self.frame_count = 0
        self.last_publish_time = time.time()
        
        # 创建定时器，用于定期处理图像和发布消息
        self.timer = self.create_timer(0.033, self.process_frame)  # 约30Hz
        
    def setup_sapien_scene(self, config_path):
        """设置 SAPIEN 场景"""
        try:
            # 初始化 SAPIEN
            sapien.render.set_viewer_shader_dir("default")
            sapien.render.set_camera_shader_dir("default")

            # 创建场景
            self.scene = sapien.Scene()
            render_mat = sapien.render.RenderMaterial()
            render_mat.base_color = [0.06, 0.08, 0.12, 1]
            render_mat.metallic = 0.0
            render_mat.roughness = 0.9
            render_mat.specular = 0.8
            self.scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

            # 设置光照
            self.scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
            self.scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
            self.scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
            self.scene.set_environment_map(
                create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
            )

            # 设置相机
            self.cam = self.scene.add_camera(
                name="Cheese!", width=800, height=600, fovy=1, near=0.1, far=10
            )
            self.cam.set_local_pose(sapien.Pose([0.30, 0, 0.0], [0, 0, 0, -1]))

            # 创建查看器
            self.viewer = Viewer()
            self.viewer.set_scene(self.scene)
            self.viewer.control_window.show_origin_frame = False
            self.viewer.control_window.move_speed = 0.01
            self.viewer.control_window.toggle_camera_lines(False)
            self.viewer.set_camera_pose(self.cam.get_local_pose())

            # 加载机器人模型
            loader = self.scene.create_urdf_loader()
            
            # 从配置文件获取 URDF 路径
            config = RetargetingConfig.load_from_file(config_path)
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

            # 加载机器人
            if "glb" not in robot_name:
                filepath = str(filepath)
            else:
                filepath = str(filepath)
            self.robot = loader.load(filepath)

            # 设置机器人位置
            if "ability" in robot_name:
                self.robot.set_pose(sapien.Pose([0, 0, -0.15]))
            elif "shadow" in robot_name:
                self.robot.set_pose(sapien.Pose([0, 0, -0.2]))
            elif "dclaw" in robot_name:
                self.robot.set_pose(sapien.Pose([0, 0, -0.15]))
            elif "allegro" in robot_name:
                self.robot.set_pose(sapien.Pose([0, 0, -0.05]))
            elif "bhand" in robot_name:
                self.robot.set_pose(sapien.Pose([0, 0, -0.2]))
            elif "leap" in robot_name:
                self.robot.set_pose(sapien.Pose([0, 0, -0.15]))
            elif "svh" in robot_name:
                self.robot.set_pose(sapien.Pose([0, 0, -0.13]))
            elif "botyard" in robot_name:
                self.robot.set_pose(sapien.Pose([0, 0, -0.2], [-0.707, 0, 0, 0.707]))  # 添加四元数旋转

            # 获取关节名称映射
            self.sapien_joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
            self.retargeting_to_sapien = np.array(
                [self.retargeting_joint_names.index(name) for name in self.sapien_joint_names]
            ).astype(int)
            
            logger.info(f"SAPIEN 场景初始化成功，使用 URDF 文件: {filepath}")
            
        except Exception as e:
            logger.error(f"SAPIEN 场景初始化失败: {e}")
            raise

    def process_frame(self):
        # 读取图像
        success, bgr = self.cap.read()
        if not success:
            return
            
        # 转换颜色空间
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # 更新帧计数
        self.frame_count += 1
        
        # 判断是否需要检测手部
        need_detection = False
        if self.last_joint_pos is None:
            need_detection = True
        elif self.hand_lost_frames >= 10:  # 连续10帧检测不到手部才重新检测
            need_detection = True
        elif self.frame_count % 5 == 0:  # 每5帧检测一次
            need_detection = True
            
        # 检测手部
        if need_detection:
            _, joint_pos, keypoint_2d, _ = self.detector.detect(rgb)
            if joint_pos is not None:
                self.last_joint_pos = joint_pos
                self.last_keypoint_2d = keypoint_2d
                self.hand_lost_frames = 0
            else:
                self.hand_lost_frames += 1
        else:
            joint_pos = self.last_joint_pos
            keypoint_2d = self.last_keypoint_2d
            
        # 绘制骨架
        if keypoint_2d is not None:
            bgr = self.detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
            
        # 显示状态信息
        status_text = f"Hand {'Lost' if self.hand_lost_frames > 0 else 'Tracked'} ({self.hand_lost_frames} frames)"
        cv2.putText(bgr, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示图像
        cv2.imshow("realtime_retargeting_demo", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.destroy_node()
            rclpy.shutdown()
            return
            
        # 控制消息发布频率（每0.1秒发布一次）
        current_time = time.time()
        if joint_pos is not None and (current_time - self.last_publish_time) >= 0.1:
            # 计算关节角度
            retargeting_type = self.retargeting.optimizer.retargeting_type
            indices = self.retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                
            qpos = self.retargeting.retarget(ref_value)
            
            # 更新机器人关节角度（仅在非无头模式下）
            if not self.headless and hasattr(self, 'robot') and hasattr(self, 'viewer'):
                try:
                    self.robot.set_qpos(qpos[self.retargeting_to_sapien])
                    # 渲染场景
                    if self.viewer is not None:
                        self.viewer.render()
                        logger.debug("渲染成功")
                except Exception as e:
                    logger.error(f"渲染错误: {e}")
            
            # 构建消息
            msg = Float64MultiArray()
            msg.layout.dim.append(MultiArrayDimension())
            msg.layout.dim.append(MultiArrayDimension())
            msg.layout.dim[0].label = "joint_list"
            msg.layout.dim[0].size = len(self.target_joint_order)
            msg.layout.dim[0].stride = len(self.target_joint_order)
            msg.layout.dim[1].label = "joint"
            msg.layout.dim[1].size = 1
            msg.layout.dim[1].stride = 1
            
            # 按照目标顺序填充数据
            msg.data = [qpos[self.retargeting_joint_names.index(name)] if name in self.retargeting_joint_names else 0.0 for name in self.target_joint_order]
            
            # 发布消息
            try:
                self.publisher.publish(msg)
                self.last_publish_time = current_time
                
                # 打印关节状态
                joint_status = "\n".join([f"{joint_name}: {msg.data[i]:.4f}" for i, joint_name in enumerate(self.target_joint_order)])
                logger.info(f"\n发布到话题的关节顺序:\n{'-' * 50}\n{joint_status}\n{'-' * 50}")
            except Exception as e:
                logger.error(f"消息发布错误: {e}")

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'viewer') and self.viewer is not None:
            try:
                self.viewer.close()  # 使用 close() 方法关闭查看器
            except:
                pass
        if hasattr(self, 'scene'):
            del self.scene


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
