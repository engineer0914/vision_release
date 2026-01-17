import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

# === 저장 폴더 ===
SAVE_DIR = "realsense_depth_saves"
os.makedirs(SAVE_DIR, exist_ok=True)

# === 파이프라인 설정 ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# === Depth scale 확인 ===
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth scale: {depth_scale} meters per unit (~{depth_scale*1000:.1f} mm/unit)")

# === Color에 Depth 정렬 ===
align_to = rs.stream.color
align = rs.align(align_to)

# === 필터 설정 ===
decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

# === 트랙바 설정 ===
window_name = 'Depth Colormap'
cv2.namedWindow(window_name)

# 최대 10 m, 최소 0.01 m
cv2.createTrackbar('MinDepth(cm)', window_name, 20, 1000, lambda x: None)
cv2.createTrackbar('MaxDepth(cm)', window_name, 500, 1000, lambda x: None)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # === 필터 적용 ===
        depth_frame = decimation.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        # === numpy 변환 ===
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # === 크기 맞추기 ===
        if depth_image.shape[:2] != color_image.shape[:2]:
            depth_image = cv2.resize(depth_image, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # === 트랙바에서 최소/최대 깊이 읽기 ===
        min_depth_cm = cv2.getTrackbarPos('MinDepth(cm)', window_name)
        max_depth_cm = cv2.getTrackbarPos('MaxDepth(cm)', window_name)
        min_depth_m = max(0.01, min_depth_cm / 100.0)
        max_depth_m = max(min_depth_m + 0.01, max_depth_cm / 100.0)  # 최소 1cm 차이 보장

        # === Depth 단위 변환 ===
        min_units = min_depth_m / depth_scale
        max_units = max_depth_m / depth_scale

        # === Depth 범위 내 mm 단위 시각화 ===
        depth_clipped = np.clip(depth_image, min_units, max_units)
        # 0~255 스케일링 (범위 내 mm 단위 반영)
        depth_normalized = ((depth_clipped - min_units) / (max_units - min_units) * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # === Overlay ===
        overlay = cv2.addWeighted(color_image, 0.6, depth_colormap, 0.4, 0)

        # === 출력 ===
        cv2.imshow('Color', color_image)
        cv2.imshow(window_name, depth_colormap)
        cv2.imshow('Overlay', overlay)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            depth_path = os.path.join(SAVE_DIR, f"depth_{ts}.png")
            color_path = os.path.join(SAVE_DIR, f"color_{ts}.png")
            cv2.imwrite(depth_path, depth_image)
            cv2.imwrite(color_path, color_image)
            print(f"Saved: {depth_path}, {color_path}")
        elif key == ord('p'):
            print(f"Depth Range: {min_depth_m:.3f} m ~ {max_depth_m:.3f} m")

except Exception as e:
    print("Error:", e)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
