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

# 패딩 값 설정
padding = 20

# 캐니 엣지 임계값 설정
canny_threshold1 = 100
canny_threshold2 = 200

# 모폴로지 연산을 위한 커널 설정
kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), np.uint8)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        h, w, _ = frame.shape

        results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                class_name = model.names[class_id]

                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(w, x2 + padding)
                y2_pad = min(h, y2 + padding)

                roi_color = frame[y1_pad:y2_pad, x1_pad:x2_pad].copy() # ROI 컬러 복사본
                
                if roi_color.shape[0] > 0 and roi_color.shape[1] > 0:
                    gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                    
                    edges = cv2.Canny(gray_roi, canny_threshold1, canny_threshold2)
                    
                    # 팽창 모폴로지 연산 적용
                    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                    
                    # 가장 바깥쪽 윤곽선만 찾기 (cv2.RETR_EXTERNAL 사용)
                    # cv2.CHAIN_APPROX_SIMPLE은 불필요한 점들을 압축하여 윤곽선 정보를 저장
                    contours, hierarchy = cv2.findContours(dilated_edges, 
                                                           cv2.RETR_EXTERNAL,  # 가장 바깥쪽 윤곽선만 찾음
                                                           cv2.CHAIN_APPROX_SIMPLE)
                    
                    # 찾은 윤곽선을 원본 컬러 ROI 이미지에 그리기
                    # -1은 모든 윤곽선을 그림, (0,255,0)은 초록색, 2는 선 두께
                    contour_image = np.zeros_like(roi_color) # 윤곽선만 그릴 빈 이미지
                    cv2.drawContours(roi_color, contours, -1, (0, 255, 0), 2)
                    # 윤곽선만 표시하고 싶다면, contour_image에 그리면 됨
                    # cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)


                    cv2.imshow(f"External Contours on ROI: {class_name}", roi_color)
                    cv2.imshow(f"Dilated Edges ROI: {class_name}", dilated_edges) # 원한다면 팽창 엣지도 계속 표시

        cv2.imshow("YOLO + RealSense", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()