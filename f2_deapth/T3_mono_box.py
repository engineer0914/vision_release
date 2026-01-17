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
        # === 프레임 데이터 가져오기 ===
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # ======================================================================
        # ★★★★★ YOLO 적용: color_image에서 객체 감지 후 ROI 가져오기 ★★★★★
        # ======================================================================
        results = model.predict(source=color_image, imgsz=640, verbose=False)

        boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
        scores = results[0].boxes.conf.cpu().numpy()

        if len(boxes) == 0:
            # 객체가 없을 때는 박스를 그리지 않고 계속 진행
            cv2.imshow('RealSense Color with Bounding Box', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 가장 확률 높은 박스 하나 선택
        best_idx = np.argmax(scores)
        x_min, y_min, x_max, y_max = boxes[best_idx].astype(int)

        padding = 10
        roi_x_min = max(0, x_min - padding)
        roi_y_min = max(0, y_min - padding)
        roi_x_max = min(color_image.shape[1], x_max + padding)
        roi_y_max = min(color_image.shape[0], y_max + padding)

        # 박스를 이미지에 표시
        cv2.rectangle(color_image,
                      (roi_x_min, roi_y_min),
                      (roi_x_max, roi_y_max),
                      (0, 255, 0), 2)
        # ======================================================================

        # === 깊이 이미지(meter 단위) ===
        depth_image_in_meters = depth_image.astype(float) * depth_scale
        height, width = depth_image.shape

        # === 2D → 3D 포인트 변환 ===
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        Z = depth_image_in_meters
        X = (u - intrinsics.ppx) * Z / intrinsics.fx
        Y = (v - intrinsics.ppy) * Z / intrinsics.fy

        points = np.dstack((X, Y, Z)).reshape(-1, 3)
        colors_bgr = color_image.reshape(-1, 3)

        # 깊이값 없는 포인트 제거
        valid_indices = np.where(points[:, 2] > 0)[0]
        points = points[valid_indices]
        colors = colors_bgr[valid_indices]
        u_flat = u.flatten()[valid_indices]
        v_flat = v.flatten()[valid_indices]

        # ======================================================================
        # ★★★★★ YOLO ROI 내부/외부 포인트 분리 ★★★★★
        # ======================================================================
        mask_roi = (u_flat >= roi_x_min) & (u_flat < roi_x_max) & \
                   (v_flat >= roi_y_min) & (v_flat < roi_y_max)

        points_roi = points[mask_roi]
        points_outside = points[~mask_roi]
        colors_outside = colors[~mask_roi] / 255.0
        # ======================================================================

        # === 성능을 위해 다운샘플링 ===
        downsample_factor = 50
        points_roi_sub = points_roi[::downsample_factor]
        points_outside_sub = points_outside[::downsample_factor]
        colors_outside_sub = colors_outside[::downsample_factor]

        # === 3D 플롯 업데이트 ===
        ax.clear()

        ax.scatter(points_outside_sub[:, 0], points_outside_sub[:, 1], points_outside_sub[:, 2],
                   c=colors_outside_sub[:, ::-1], s=0.1)

        ax.scatter(points_roi_sub[:, 0], points_roi_sub[:, 1], points_roi_sub[:, 2],
                   c='red', s=0.5)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Point Cloud with YOLO ROI Highlighted')
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(0.3, 0.5)
        ax.invert_zaxis()

        fig.canvas.draw()
        plt.pause(0.001)

        # === OpenCV GUI 출력 ===
        cv2.imshow('RealSense Color with Bounding Box', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("스트리밍을 종료합니다...")
    pipeline.stop()
    cv2.destroyAllWindows()
    plt.ioff()
    print("모든 창이 닫혔습니다.")

