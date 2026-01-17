import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------- YOLO 모델 로드 -----------------------
from ultralytics import YOLO
model = YOLO("./runs/detect/train_8n/weights/best.pt")
print("YOLO 모델을 성공적으로 불러왔습니다.")

# ------------------------------------------------------------------------------------------
# 1. RealSense 파이프라인 및 카메라 설정 초기화
# ------------------------------------------------------------------------------------------
print("RealSense 파이프라인을 설정합니다...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# ------------------------------------------------------------------------------------------
# 2. Matplotlib 3D 시각화 준비
# ------------------------------------------------------------------------------------------
print("Matplotlib 시각화 창을 준비합니다...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.ion()
fig.show()
fig.canvas.draw()

# ------------------------------------------------------------------------------------------
# 3. 메인 루프
# ------------------------------------------------------------------------------------------
print("스트리밍 및 렌더링을 시작합니다. 'q' 키를 눌러 종료하세요.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # ======================================================================
        # ★★★★★ YOLO 적용: 여러 객체 감지 처리 ★★★★★
        # ======================================================================
        results = model.predict(source=color_image, imgsz=640, verbose=False)

        boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
        scores = results[0].boxes.conf.cpu().numpy()

        # YOLO 박스가 없다면 그대로 continue
        if len(boxes) == 0:
            cv2.imshow('RealSense Color with Bounding Box', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ROI 목록 생성
        roi_list = []

        for (x_min, y_min, x_max, y_max) in boxes.astype(int):
            padding = 10
            roi_x_min = max(0, x_min - padding)
            roi_y_min = max(0, y_min - padding)
            roi_x_max = min(color_image.shape[1], x_max + padding)
            roi_y_max = min(color_image.shape[0], y_max + padding)

            roi_list.append((roi_x_min, roi_y_min, roi_x_max, roi_y_max))

            # 각각의 박스를 그림
            cv2.rectangle(color_image,
                          (roi_x_min, roi_y_min),
                          (roi_x_max, roi_y_max),
                          (0, 255, 0), 2)

        # ======================================================================

        # === 깊이(m) 변환 ===
        depth_image_in_meters = depth_image.astype(float) * depth_scale
        height, width = depth_image.shape

        # === 2D to 3D ===
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        Z = depth_image_in_meters
        X = (u - intrinsics.ppx) * Z / intrinsics.fx
        Y = (v - intrinsics.ppy) * Z / intrinsics.fy

        points = np.dstack((X, Y, Z)).reshape(-1, 3)
        colors_bgr = color_image.reshape(-1, 3)

        valid_indices = np.where(points[:, 2] > 0)[0]
        points = points[valid_indices]
        colors = colors_bgr[valid_indices]
        u_flat = u.flatten()[valid_indices]
        v_flat = v.flatten()[valid_indices]

        # ======================================================================
        # ★★★★★ 여러 ROI를 모두 포함한 전체 마스크 생성 ★★★★★
        # ======================================================================
        total_mask = np.zeros(len(points), dtype=bool)

        for (x1, y1, x2, y2) in roi_list:
            mask_roi = (u_flat >= x1) & (u_flat < x2) & \
                       (v_flat >= y1) & (v_flat < y2)
            total_mask |= mask_roi  # OR 조건 → 여러 영역 합치기

        points_roi = points[total_mask]
        points_outside = points[~total_mask]
        colors_outside = colors[~total_mask] / 255.0
        # ======================================================================

        # === 다운샘플링 ===
        downsample = 50
        points_roi_sub = points_roi[::downsample]
        points_outside_sub = points_outside[::downsample]
        colors_outside_sub = colors_outside[::downsample]

        # === 3D 갱신 ===
        ax.clear()

        # 외부 포인트
        ax.scatter(points_outside_sub[:, 0], points_outside_sub[:, 1], points_outside_sub[:, 2],
                   c=colors_outside_sub[:, ::-1], s=0.1)

        # ROI 포인트(모든 박스) - 빨간색
        ax.scatter(points_roi_sub[:, 0], points_roi_sub[:, 1], points_roi_sub[:, 2],
                   c='red', s=0.5)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Point Cloud with Multiple YOLO ROIs Highlighted')
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(0.3, 0.5)
        ax.invert_zaxis()

        fig.canvas.draw()
        plt.pause(0.001)

        cv2.imshow('RealSense Color with Bounding Box', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("스트리밍을 종료합니다...")
    pipeline.stop()
    cv2.destroyAllWindows()
    plt.ioff()
    print("모든 창이 닫혔습니다.")
