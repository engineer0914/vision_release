import pyrealsense2 as rs
import numpy as np
import cv2

# --- 파이프라인 설정 ---
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = any(s.get_info(rs.camera_info.name) == 'RGB Camera' for s in device.sensors)
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# --- 스트리밍 시작 ---
pipeline.start(config)

# --- 트랙바 설정 ---
window_name = 'RealSense'
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

# alpha 조정 범위 (0~100까지, 내부적으로 0.01~1.0으로 매핑)
cv2.createTrackbar('alpha', window_name, 10, 100, lambda x: None)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # --- 트랙바에서 alpha 값 읽기 ---
        alpha_raw = cv2.getTrackbarPos('alpha', window_name)
        alpha = alpha_raw / 1000.0  # 0~100 → 0~2.0 사이 값 (조정 가능)

        # --- 깊이 영상 컬러맵 ---
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=alpha), cv2.COLORMAP_JET
        )

        # --- 크기 맞추기 ---
        if depth_colormap.shape != color_image.shape:
            color_resized = cv2.resize(color_image, (depth_colormap.shape[1], depth_colormap.shape[0]))
            images = np.hstack((color_resized, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # --- 표시 ---
        cv2.imshow(window_name, images)
        key = cv2.waitKey(1)
        if key == 27:  # ESC 누르면 종료
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
