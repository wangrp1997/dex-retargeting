import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional
import yaml
import datetime

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



def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    # Setup SAPIEN scene
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

    if "glb" not in robot_name:
        filepath = str(filepath)
    else:
        filepath = str(filepath)

    robot = loader.load(filepath)

    # 设置机器人初始位置
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

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]

    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    # 如果是 DEXPILOT 方法，先进行标定
    if retargeting.optimizer.retargeting_type == "DEXPILOT":
        logger.info("Starting calibration...")
        logger.info("Please place your hand 30-50cm in front of the camera, keep your palm open")
        cv2.namedWindow("calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("calibration", 800, 600)
        
        # Set initial joint angles for calibration (open palm state)
        if "botyard" in robot_name:
            # 打印关节名称和索引的对应关系
            logger.info("=== 关节名称和索引对应关系 ===")
            logger.info("机器人所有关节:")
            for i, name in enumerate(retargeting.optimizer.robot.dof_joint_names):
                logger.info(f"{i}: {name}")
            
            logger.info("\n目标关节:")
            for i, name in enumerate(retargeting.optimizer.target_joint_names):
                idx = retargeting.optimizer.idx_pin2target[i]
                logger.info(f"{i} -> {idx}: {name}")
            
            logger.info("\nSAPIEN关节:")
            for i, name in enumerate(sapien_joint_names):
                idx = retargeting_to_sapien[i]
                logger.info(f"{i} -> {idx}: {name}")
            
            # Set initial joint angles for open palm state
            init_qpos = np.zeros(retargeting.optimizer.num_joints, dtype=np.float32)
            
            # 获取关节极限
            joint_limits = np.stack([
                retargeting.optimizer.opt.get_lower_bounds(),
                retargeting.optimizer.opt.get_upper_bounds()
            ], axis=1)
            
            # 根据关节名称设置角度
            joint_angles = {
                # 手掌关节
                "FAJ3": 0.0,  # 手掌俯仰
                "FAJ1": 0.0,  # 手掌偏航
                
                # 拇指关节
                "THJ4": 0.0,  # 拇指基座
                "THJ3": 0.0,  # 拇指近端
                "THJ2": 0.0,  # 拇指中间
                "THJ1": 0.0,  # 拇指远端
                
                # 食指关节
                "FFJ4": joint_limits[retargeting.optimizer.target_joint_names.index("FFJ4"), 1],  # 食指基座
                "FFJ3": 0.0,  # 食指近端
                "FFJ2": 0.0,  # 食指中间
                "FFJ1": 0.0,  # 食指远端
                
                # 中指关节
                "MFJ4": 0.0,  # 中指基座
                "MFJ3": 0.0,  # 中指近端
                "MFJ2": 0.0,  # 中指中间
                "MFJ1": 0.0,  # 中指远端
                
                # 无名指关节
                "RFJ4": -0.06545,  # 无名指基座
                "RFJ3": 0.0,  # 无名指近端
                "RFJ2": 0.0,  # 无名指中间
                "RFJ1": 0.0,  # 无名指远端
                
                # 小指关节
                "LFJ5": 0.0,  # 小指手掌
                "LFJ4": joint_limits[retargeting.optimizer.target_joint_names.index("LFJ4"), 0],  # 小指基座
                "LFJ3": 0.0,  # 小指近端
                "LFJ2": 0.0,  # 小指中间
                "LFJ1": 0.0,  # 小指远端
            }
            
            # 根据关节名称设置角度
            for i, joint_name in enumerate(retargeting.optimizer.target_joint_names):
                if joint_name in joint_angles:
                    idx = retargeting.optimizer.idx_pin2target[i]
                    init_qpos[idx] = joint_angles[joint_name]
                    logger.info(f"设置关节 {joint_name} (索引 {idx}) 的角度为 {joint_angles[joint_name]}")
            
            # Set robot joint angles according to retargeting_to_sapien mapping
            robot.set_qpos(init_qpos[retargeting_to_sapien])
            # Synchronize optimizer.robot joint angles
            retargeting.optimizer.robot.compute_forward_kinematics(init_qpos)
        
        while True:
            try:
                bgr = queue.get(timeout=5)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Empty:
                logger.error("Failed to get camera image")
                return
                
            # Detect hand
            _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
            
            if joint_pos is not None:
                # Calculate human hand vectors
                indices = retargeting.optimizer.target_link_human_indices
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                human_vectors = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                human_scales = np.linalg.norm(human_vectors, axis=1)
                
                # Calculate robot vectors using optimizer.robot
                qpos = robot.get_qpos()
                retargeting.optimizer.robot.compute_forward_kinematics(qpos)
                
                # Get robot link positions
                target_link_poses = [
                    retargeting.optimizer.robot.get_link_pose(index) 
                    for index in retargeting.optimizer.computed_link_indices
                ]
                robot_pos = np.array([pose[:3, 3] for pose in target_link_poses])
                
                # Calculate robot vectors
                origin_link_pos = robot_pos[retargeting.optimizer.origin_link_indices]
                task_link_pos = robot_pos[retargeting.optimizer.task_link_indices]
                robot_vectors = task_link_pos - origin_link_pos
                robot_scales = np.linalg.norm(robot_vectors, axis=1)
                
                # Calculate scaling factors for each vector
                scaling_factors = robot_scales / human_scales
                # Set all scaling factors in optimizer
                retargeting.optimizer.scaling = scaling_factors
                print("scaling_factors",scaling_factors)
                
                # Log vector information for debugging
                logger.info("\n=== Vector Scaling Information ===")
                for i in range(len(scaling_factors)):
                    origin_link = retargeting.optimizer.robot.link_names[retargeting.optimizer.computed_link_indices[retargeting.optimizer.origin_link_indices[i]]]
                    target_link = retargeting.optimizer.robot.link_names[retargeting.optimizer.computed_link_indices[retargeting.optimizer.task_link_indices[i]]]
                    logger.info(f"\nRobot vector: {origin_link} -> {target_link} Vector scaling: {scaling_factors[i]:.3f}")
                
                # Display calibration information
                bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
                cv2.putText(bgr, f"Scaling factors: {scaling_factors.mean():.2f} (mean)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(bgr, "Press SPACE to confirm calibration", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("calibration", bgr)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space key to confirm calibration
                    cv2.destroyWindow("calibration")
                    logger.info(f"Calibration completed, scaling factors: {scaling_factors.tolist()}")
                    
                    # Save calibration data
                    calibration_data = {
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        'robot_name': robot_name,
                        'hand_type': hand_type,
                        'retargeting_type': retargeting.optimizer.retargeting_type,
                        'scaling_factors': scaling_factors.tolist(),  # 保存所有缩放因子
                        'initial_joint_angles': {
                            name: float(angle) for name, angle in zip(retargeting.joint_names, init_qpos)
                        },
                        'human_scale': float(human_scales.mean()),
                        'robot_scale': float(robot_scales.mean()),
                        'joint_names': retargeting.joint_names,
                        'target_link_human_indices': retargeting.optimizer.target_link_human_indices.tolist(),
                        'computed_link_indices': retargeting.optimizer.computed_link_indices,
                        'origin_link_indices': retargeting.optimizer.origin_link_indices.tolist(),
                        'task_link_indices': retargeting.optimizer.task_link_indices.tolist()
                    }
                    
                    # Create calibration directory if it doesn't exist
                    calibration_dir = Path(__file__).parent / "calibration_data"
                    calibration_dir.mkdir(exist_ok=True)
                    
                    # Save to YAML file
                    calibration_file = calibration_dir / f"calibration_{robot_name}_{hand_type}_{calibration_data['timestamp']}.yaml"
                    with open(calibration_file, 'w') as f:
                        yaml.dump(calibration_data, f, default_flow_style=False)
                    
                    logger.info(f"Calibration data saved to: {calibration_file}")
                    break
            else:
                cv2.putText(bgr, "No hand detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("calibration", bgr)
                cv2.waitKey(1)
            
            # Render SAPIEN scene
            viewer.render()

    # 添加手部状态跟踪变量
    last_joint_pos = None
    last_keypoint_2d = None
    hand_lost_frames = 0
    HAND_LOST_THRESHOLD = 10  # 连续多少帧检测不到手部才重新检测
    DETECTION_INTERVAL = 5    # 正常检测的间隔帧数
    frame_count = 0

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
            
            fixed_qpos = np.zeros(3)  # 3个关节都设为 0
            qpos = retargeting.retarget(ref_value, fixed_qpos=fixed_qpos)
            robot.set_qpos(qpos[retargeting_to_sapien])
            
            # 打印关节名称和对应的 qpos 值
            # print("\n当前关节状态:")
            # print("-" * 50)
            # for joint_name in retargeting.joint_names:
            #     idx = retargeting_joint_names.index(joint_name)
            #     print(f"{joint_name}: {qpos[idx]:.4f}")
            # print("-" * 50)

        # 减少渲染次数
        viewer.render()


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


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    queue = multiprocessing.Queue(maxsize=1000)
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, camera_path)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting, args=(queue, str(robot_dir), str(config_path))
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
