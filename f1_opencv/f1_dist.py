import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("./runs/detect/train_8n/weights/best.pt")

# RealSense 파이프라인
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 패딩 값 설정 (원하는 픽셀 수로 조절)
padding = 20 # 예를 들어, 상하좌우 20픽셀씩 확장

# 캐니 엣지 임계값 설정 (필요에 따라 조절)
canny_threshold1 = 100
canny_threshold2 = 200

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        h, w, _ = frame.shape # 프레임의 높이와 너비 얻기

        results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                class_name = model.names[class_id]

                # 패딩 적용된 새로운 좌표 계산
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(w, x2 + padding)
                y2_pad = min(h, y2 + padding)

                # 패딩 적용된 ROI 추출
                roi_for_canny = frame[y1_pad:y2_pad, x1_pad:x2_pad]

                if roi_for_canny.shape[0] > 0 and roi_for_canny.shape[1] > 0:
                    # ROI를 그레이스케일로 변환
                    gray_roi = cv2.cvtColor(roi_for_canny, cv2.COLOR_BGR2GRAY)
                    
                    # 캐니 엣지 적용
                    edges = cv2.Canny(gray_roi, canny_threshold1, canny_threshold2)
                    
                    # 엣지 이미지 표시
                    cv2.imshow(f"Canny Edges ROI: {class_name}", edges)

                    # --- 흰색 마스크 생성 ---
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    white_mask = np.zeros_like(gray_roi)
                    cv2.drawContours(white_mask, contours, -1, 255, cv2.FILLED)
                    cv2.imshow(f"White Mask ROI: {class_name}", white_mask)

                    # --- 거리 변환 적용 ---
                    # 거리 변환은 배경이 검은색(0)이고 객체가 흰색(255)인 2진 이미지에서 가장 잘 작동
                    # white_mask는 이미 이 형식에 적합
                    
                    # cv2.DIST_L2: 유클리드 거리 (가장 일반적)
                    # 5: 마스크 크기 (3x3, 5x5 등, cv2.DIST_MASK_5 또는 cv2.DIST_MASK_PRECISE)
                    distance_transform = cv2.distanceTransform(white_mask, cv2.DIST_L2, 5)
                    
                    # 거리 변환 결과는 float 타입이므로 시각화를 위해 정규화
                    # 0-255 범위로 스케일링하여 8비트 이미지로 변환
                    normalized_distance_transform = cv2.normalize(distance_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                    
                    # 거리 변환 결과 표시
                    cv2.imshow(f"Distance Transform ROI: {class_name}", normalized_distance_transform)

        cv2.imshow("YOLO + RealSense", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()