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
                ('camera_path', 'none')
            ]
        )
        
        # 获取参数
        robot_name = RobotName[self.get_parameter('robot_name').value.lower()]
        retargeting_type = RetargetingType[self.get_parameter('retargeting_type').value.lower()]
        hand_type = HandType[self.get_parameter('hand_type').value.lower()]
        camera_path = self.get_parameter('camera_path').value
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
        
        # 定义目标关节顺序
        self.target_joint_order = [
            'FAJ1', 'FAJ3', 'FFJ1', 'FFJ2', 'FFJ3', 'FFJ4',
            'LFJ1', 'LFJ2', 'LFJ3', 'LFJ4', 'LFJ5',
            'MFJ1', 'MFJ2', 'MFJ3', 'MFJ4',
            'RFJ1', 'RFJ2', 'RFJ3', 'RFJ4',
            'THJ1', 'THJ2', 'THJ3', 'THJ4'
        ]
        
        # 获取关节名称映射
        self.retargeting_joint_names = self.retargeting.joint_names
        self.retargeting_to_sapien = np.array(
            [self.retargeting_joint_names.index(name) for name in self.target_joint_order]
        ).astype(int)
        
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
            
            # 创建关节角度映射
            joint_angles = {}
            for i, joint_name in enumerate(self.target_joint_order):
                joint_angles[joint_name] = qpos[self.retargeting_to_sapien][i]
                
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
            msg.data = [joint_angles.get(joint_name, 0.0) for joint_name in self.target_joint_order]
            
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
