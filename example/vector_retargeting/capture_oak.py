from pathlib import Path
from typing import Union
import cv2
import tyro
from oak import OakCamera

def main(
    video_path: str,
    resolution: str = "full",
    internal_fps: int = None,
    save_depth: bool = False,
    crop: bool = False,
    internal_frame_height: int = 640
):
    """
    使用 OAK-D 相机录制视频。按 'ESC' 键结束录制。

    Args:
        video_path: 输出视频文件的路径（.mp4格式）
        resolution: 分辨率选择 "full" (1280x800) 或 "ultra" (3840x2160)
        internal_fps: 相机帧率，如果不指定则使用默认值
        save_depth: 是否保存深度数据
        crop: 是否裁剪图像为正方形
        internal_frame_height: 内部处理的图像高度
    """
    # 初始化 OAK-D 相机
    cap = OakCamera(
        input_src="rgb",
        resolution=resolution,
        internal_fps=internal_fps,
        xyz=save_depth,
        crop=crop,
        internal_frame_height=internal_frame_height
    )
    
    # 获取第一帧以确定视频尺寸
    ret, frame, depth = cap.read()
    if not ret:
        print("无法从相机获取图像")
        return
        
    height, width = frame.shape[:2]
    
    # 创建视频写入器
    path = Path(video_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    writer_rgb = cv2.VideoWriter(
        str(video_path), 
        cv2.VideoWriter_fourcc(*"mp4v"), 
        cap.internal_fps, 
        (width, height)
    )

    if save_depth and depth is not None:
        depth_path = path.with_name(path.stem + "_depth.mp4")
        depth_h, depth_w = depth.shape
        writer_depth = cv2.VideoWriter(
            str(depth_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            cap.internal_fps,
            (depth_w, depth_h),
            isColor=False  # 指定为单通道
        )
    else:
        writer_depth = None

    while True:
        ret, frame, depth = cap.read()
        if not ret:
            break
            
        writer_rgb.write(frame)
        if writer_depth is not None:
            # 将深度数据归一化到 0-255 范围以便保存
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            writer_depth.write(depth_normalized)
        
        # 显示图像
        cv2.imshow("RGB", frame)
        if depth is not None:
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            cv2.imshow("Depth", depth_colormap)
        
        # 按 ESC 键退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

    print("Recording finished")
    cap.release()
    writer_rgb.release()
    if writer_depth is not None:
        writer_depth.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    tyro.cli(main)