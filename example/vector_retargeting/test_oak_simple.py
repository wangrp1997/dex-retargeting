import cv2
import time
from oak import OakCamera

print("开始初始化相机...")

try:
    # 使用官方 OakCamera 类
    cap = OakCamera(
        input_src="rgb",
        resolution="full",  # 使用 1280x800 分辨率
        internal_fps=30,
        xyz=False,  # 不使用深度相机
        crop=False,
        internal_frame_height=640
    )
    print("相机初始化完成")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        try:
            # 读取一帧
            ret, frame, _ = cap.read()
            if not ret:
                print("读取帧失败")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            if frame_count % 30 == 0:  # 每30帧打印一次状态
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"已处理 {frame_count} 帧，FPS: {fps:.2f}")
            
            # 显示图像
            cv2.imshow("OAK-D Test", frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
                
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
            time.sleep(0.1)
            continue
            
except Exception as e:
    print(f"相机初始化失败: {str(e)}")
    import traceback
    traceback.print_exc()
finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("程序结束") 