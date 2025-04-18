import depthai as dai
import cv2
import numpy as np
import mediapipe_utils as mpu
from pathlib import Path
from FPS import FPS, now

SCRIPT_DIR = Path(__file__).resolve().parent

class OakCamera:
    """OAK-D 相机类"""
    def __init__(self, 
                 input_src=None,
                 xyz=True,
                 crop=False,
                 internal_fps=None,
                 resolution="full",
                 internal_frame_height=640):
                 
        self.device = dai.Device()
        
        if input_src == None or input_src == "rgb" or input_src == "rgb_laconic":
            self.input_type = "rgb"
            self.laconic = input_src == "rgb_laconic"
            
            if resolution == "full":
                self.resolution = (1280, 800)
            elif resolution == "ultra":
                self.resolution = (3840, 2160)
            else:
                print(f"Error: {resolution} is not a valid resolution !")
                sys.exit()
            print("Sensor resolution:", self.resolution)

            # 检查深度功能可用性
            if xyz:
                cameras = self.device.getConnectedCameras()
                if dai.CameraBoardSocket.CAM_B in cameras and dai.CameraBoardSocket.CAM_C in cameras:
                    self.xyz = True
                else:
                    print("Warning: depth unavailable on this device, 'xyz' argument is ignored")
                    self.xyz = False
            else:
                self.xyz = False

            # 设置 FPS
            if internal_fps is None:
                if self.xyz:
                    self.internal_fps = 30
                else:
                    self.internal_fps = 40
            else:
                self.internal_fps = internal_fps
            print(f"Internal camera FPS set to: {self.internal_fps}")

            self.video_fps = self.internal_fps
            self.crop = crop
            # 设置图像尺寸和裁剪参数
            if self.crop:
                self.frame_size, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height, self.resolution)
                self.img_h = self.img_w = self.frame_size
                self.pad_w = self.pad_h = 0
                self.crop_w = (int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1])) - self.img_w) // 2
            else:
                width, self.scale_nd = mpu.find_isp_scale_params(
                    internal_frame_height * self.resolution[0] / self.resolution[1], 
                    self.resolution, 
                    is_height=False
                )
                self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
                self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
                self.pad_h = (self.img_w - self.img_h) // 2
                self.pad_w = 0
                self.frame_size = self.img_w
                self.crop_w = 0

            print(f"Internal camera image size: {self.img_w} x {self.img_h} - pad_h: {self.pad_h}")
        
        else:
            print("Invalid input source:", input_src)
            sys.exit()

        # 创建并启动管线
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")

        # 创建输出队列
        if not self.laconic:
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
        if self.xyz:
            self.q_depth = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)

        self.fps = FPS()

    def create_pipeline(self):
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)

        # 创建彩色相机节点
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        if self.resolution[0] == 1280:
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
        else:
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        cam.setInterleaved(False)
        cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
        cam.setFps(self.internal_fps)

        if self.crop:
            cam.setVideoSize(self.frame_size, self.frame_size)
            cam.setPreviewSize(self.frame_size, self.frame_size)
        else:
            cam.setVideoSize(self.img_w, self.img_h)
            cam.setPreviewSize(self.img_w, self.img_h)

        if not self.laconic:
            cam_out = pipeline.createXLinkOut()
            cam_out.setStreamName("cam_out")
            cam_out.input.setQueueSize(1)
            cam_out.input.setBlocking(False)
            cam.video.link(cam_out.input)

        # 设置深度相机
        if self.xyz:
            print("Creating MonoCameras, Stereo and SpatialLocationCalculator nodes...")
            # For now, RGB needs fixed focus to properly align with depth.
            # The value used during calibration should be used here
            # calib_data = self.device.readCalibration()
            # calib_lens_pos = calib_data.getLensPosition(dai.CameraBoardSocket.CAM_B)
            # print(f"RGB calibration lens position: {calib_lens_pos}")
            # cam.initialControl.setManualFocus(calib_lens_pos)

            left = cam

            right = pipeline.createColorCamera()
            right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
            right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
            right.setInterleaved(False)
            right.setIspScale(self.scale_nd[0], self.scale_nd[1])
            right.setFps(self.internal_fps)

            if self.crop:
                cam.setVideoSize(self.frame_size, self.frame_size)
                cam.setPreviewSize(self.frame_size, self.frame_size)
            else: 
                cam.setVideoSize(self.img_w, self.img_h)
                cam.setPreviewSize(self.img_w, self.img_h)

            stereo = pipeline.createStereoDepth()
            stereo.setConfidenceThreshold(230)
            # LR-check is required for depth alignment
            stereo.setLeftRightCheck(True)
            # stereo.setDepthAlign(dai.CameraBoardSocket.CAM_C)
            stereo.setSubpixel(False)  # subpixel True brings latency
            # MEDIAN_OFF necessary in depthai 2.7.2. 
            # Otherwise : [critical] Fatal error. Please report to developers. Log: 'StereoSipp' '533'
            # stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)

            spatial_location_calculator = pipeline.createSpatialLocationCalculator()
            spatial_location_calculator.setWaitForConfigInput(True)
            spatial_location_calculator.inputDepth.setBlocking(False)
            spatial_location_calculator.inputDepth.setQueueSize(1)

            left.isp.link(stereo.left)
            right.isp.link(stereo.right)    

            stereo.depth.link(spatial_location_calculator.inputDepth)

            # 创建深度输出
            xout_depth = pipeline.createXLinkOut()
            xout_depth.setStreamName("depth")
            stereo.depth.link(xout_depth.input)

        return pipeline

    def read(self):
        """读取一帧图像和深度数据（如果可用）"""
        self.fps.update()
        
        if self.laconic:
            video_frame = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        else:
            in_video = self.q_video.get()
            video_frame = in_video.getCvFrame()

        if self.xyz:
            depth_data = self.q_depth.get()
            depth_frame = depth_data.getCvFrame()
            return True, video_frame, depth_frame
        else:
            return True, video_frame, None

    def release(self):
        """释放设备"""
        self.device.close()
        print(f"FPS : {self.fps.get_global():.1f} f/s (# frames = {self.fps.nb_frames()})")