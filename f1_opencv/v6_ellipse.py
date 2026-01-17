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

# 윤곽선 최소 길이 (타원을 그리기 위한 최소 점 개수 5개 이상)
min_contour_length_for_ellipse = 5

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

                roi_color = frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()
                
                if roi_color.shape[0] > 0 and roi_color.shape[1] > 0:
                    gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                    
                    edges = cv2.Canny(gray_roi, canny_threshold1, canny_threshold2)
                    
                    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                    
                    contours, hierarchy = cv2.findContours(dilated_edges, 
                                                           cv2.RETR_EXTERNAL, 
                                                           cv2.CHAIN_APPROX_SIMPLE)
                    
                    # 가장 큰(바깥쪽) 윤곽선을 찾아서 타원 그리기
                    if contours:
                        # 윤곽선 중 가장 큰 면적을 가진 윤곽선을 찾거나, 단순히 첫 번째 윤곽선을 사용
                        # 여기서는 가장 큰 면적의 윤곽선을 사용해 객체의 주 윤곽선을 가정합니다.
                        largest_contour = max(contours, key=cv2.contourArea)

                        # 타원을 그리기 위한 최소 점 개수 확인
                        if len(largest_contour) >= min_contour_length_for_ellipse:
                            # 윤곽선에 가장 잘 맞는 타원 계산
                            ellipse = cv2.fitEllipse(largest_contour)
                            
                            # 계산된 타원을 ROI 이미지에 그림
                            # 타원 색상 (빨간색), 두께 2
                            cv2.ellipse(roi_color, ellipse, (0, 0, 255), 2)
                        else:
                            # 타원을 그릴 수 없는 경우, 그냥 윤곽선만 그림
                            cv2.drawContours(roi_color, [largest_contour], -1, (0, 255, 0), 2)
                    
                    cv2.imshow(f"Ellipse on ROI: {class_name}", roi_color)

        cv2.imshow("YOLO + RealSense", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()