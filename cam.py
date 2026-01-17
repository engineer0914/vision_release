import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("/home/user/Desktop/yolov11/runs/detect/train_8n/weights/best.pt")

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
        results = model.predict(frame, imgsz=640, conf=0.3, verbose=False)

        # 결과 그리기
        annotated_frame = results[0].plot()

        cv2.imshow("YOLO + RealSense", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
