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

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:   # 프레임 못 받았으면 건너뛰기
            continue

        frame = np.asanyarray(color_frame.get_data())

        # YOLO 추론 (show=True 대신 결과를 직접 표시)
        results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)

        # 결과 그리기
        annotated_frame = results[0].plot()

        # 각 감지된 객체에 대해 ROI 창 생성
        for r in results:
            for box in r.boxes:
                # 바운딩 박스 좌표 얻기
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 클래스 이름 얻기
                class_id = int(box.cls)
                class_name = model.names[class_id]

                # ROI 추출
                roi = frame[y1:y2, x1:x2]

                # ROI 창 표시 (창 이름은 감지된 물체 이름)
                if roi.shape[0] > 0 and roi.shape[1] > 0: # ROI가 비어있지 않은지 확인
                    cv2.imshow(f"ROI: {class_name}", roi)

        cv2.imshow("YOLO + RealSense", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()