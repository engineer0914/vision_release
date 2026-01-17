import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
####수정 필요
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
        
        # ---------------------------------------------------------------------------- #
        # ★★★★★ YOLO 시뮬레이션 영역 ★★★★★
        # ---------------------------------------------------------------------------- #
        # 실제로는 이 부분에 YOLO 모델을 실행하여 bounding box를 얻는 코드가 들어갑니다.
        # 여기서는 화면 중앙에 가상의 박스가 있다고 가정하겠습니다.
        
        # 가상 바운딩 박스 (x_min, y_min, x_max, y_max)
        x_min, y_min = 220, 140
        x_max, y_max = 420, 340
        padding = 10 # 바운딩 박스에 여백 추가
        
        # 패딩을 적용한 최종 ROI 좌표
        roi_x_min = max(0, x_min - padding)
        roi_y_min = max(0, y_min - padding)
        roi_x_max = min(color_image.shape[1], x_max + padding)
        roi_y_max = min(color_image.shape[0], y_max + padding)
        
        # OpenCV 이미지에 바운딩 박스를 그려서 확인
        cv2.rectangle(color_image, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (0, 255, 0), 2)
        # ---------------------------------------------------------------------------- #

        depth_image_in_meters = depth_image.astype(float) * depth_scale
        height, width = depth_image.shape
        
        # === Numpy를 이용한 2D -> 3D 변환 ===
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        Z = depth_image_in_meters
        X = (u - intrinsics.ppx) * Z / intrinsics.fx
        Y = (v - intrinsics.ppy) * Z / intrinsics.fy
        
        points = np.dstack((X, Y, Z)).reshape(-1, 3)
        colors_bgr = color_image.reshape(-1, 3)
        
        # 유효하지 않은 점들(깊이 값이 0인 점들) 제거
        valid_indices = np.where(points[:, 2] > 0)[0]
        points = points[valid_indices]
        colors = colors_bgr[valid_indices]
        # u, v 좌표도 유효한 점들에 대해서만 남겨야 마스크와 길이가 맞습니다.
        u_flat = u.flatten()[valid_indices]
        v_flat = v.flatten()[valid_indices]

        # === ★★★★★ 바운딩 박스 내부/외부 포인트 분리 ★★★★★ ===
        # 1. 1D 마스크 생성: 각 포인트가 ROI 내부에 있는지 여부를 나타내는 boolean 배열
        mask_roi = (u_flat >= roi_x_min) & (u_flat < roi_x_max) & \
                   (v_flat >= roi_y_min) & (v_flat < roi_y_max)
        
        # 2. 마스크를 이용해 포인트와 색상을 두 그룹으로 분리
        points_roi = points[mask_roi]      # ROI 내부의 3D 좌표
        colors_roi_original = colors[mask_roi] # (나중에 필요하면 사용)

        points_outside = points[~mask_roi] # ROI 외부의 3D 좌표 (~는 NOT 연산자)
        colors_outside = colors[~mask_roi] # ROI 외부의 색상
        
        # === 성능을 위한 다운샘플링 ===
        downsample_factor = 50
        
        points_roi_sub = points_roi[::downsample_factor]
        
        points_outside_sub = points_outside[::downsample_factor]
        colors_outside_sub = colors_outside[::downsample_factor] / 255.0

        # === 3D 플롯 업데이트 (두 그룹으로 나누어 그리기) ===
        ax.clear()
        
        # 1. ROI 외부의 점들을 원래 색상으로 그리기
        ax.scatter(points_outside_sub[:, 0], points_outside_sub[:, 1], points_outside_sub[:, 2], 
                   c=colors_outside_sub[:, ::-1], # BGR -> RGB
                   s=0.1)
                   
        # 2. ROI 내부의 점들을 지정된 색상(빨간색)으로 그리기
        #    색상을 단일 값으로 지정하면 모든 점이 해당 색상으로 그려집니다.
        ax.scatter(points_roi_sub[:, 0], points_roi_sub[:, 1], points_roi_sub[:, 2], 
                   c='red', # 색상을 빨간색으로 지정
                   s=0.5)   # 눈에 잘 띄도록 점 크기를 약간 키움

        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title('Point Cloud with YOLO ROI Highlighted')
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(0.3, 0.5)
        ax.invert_zaxis()

        fig.canvas.draw()
        plt.pause(0.001)
        
        # === OpenCV로 2D 이미지 창 표시 ===
        cv2.imshow('RealSense Color with Bounding Box', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # ------------------------------------------------------------------------------------------
    # 4. 종료 및 리소스 정리
    # ------------------------------------------------------------------------------------------
    print("스트리밍을 종료합니다...")
    pipeline.stop()
    cv2.destroyAllWindows()
    plt.ioff()
    print("모든 창이 닫혔습니다.")