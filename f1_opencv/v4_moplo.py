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

# 모폴로지 연산을 위한 커널 설정
# 커널 크기를 조절하여 모폴로지 효과를 변경할 수 있습니다.
kernel_size = 2
kernel = np.ones((kernel_size, kernel_size), np.uint8)

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
                roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]

                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    # 캐니 엣지 적용
                    edges = cv2.Canny(gray_roi, canny_threshold1, canny_threshold2)
                    
                    # 팽창(Dilate) 모폴로지 연산 적용
                    # 엣지를 더 두껍게 만들거나 끊어진 부분을 연결하는 효과
                    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                    
                    # 팽창된 엣지 이미지 표시
                    cv2.imshow(f"Dilated Edges ROI: {class_name}", dilated_edges)

                    # 필요하다면 원본 캐니 엣지 이미지도 계속 표시
                    cv2.imshow(f"Canny Edges ROI: {class_name}", edges)

        cv2.imshow("YOLO + RealSense", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()