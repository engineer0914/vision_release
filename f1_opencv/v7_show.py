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
        annotated_frame = results[0].plot() # YOLOv8의 plot 함수가 바운딩 박스와 라벨을 그려줌

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
                    
                    rotation_angle = None # 회전 각도 초기화
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)

                        if len(largest_contour) >= min_contour_length_for_ellipse:
                            ellipse = cv2.fitEllipse(largest_contour)
                            rotation_angle = ellipse[2] # 회전 각도 저장
                            
                            cv2.ellipse(roi_color, ellipse, (0, 0, 255), 2)
                        else:
                            cv2.drawContours(roi_color, [largest_contour], -1, (0, 255, 0), 2)
                    
                    cv2.imshow(f"Ellipse on ROI: {class_name}", roi_color)

                # 메인 annotated_frame에 텍스트 추가
                if rotation_angle is not None:
                    # 표시할 텍스트
                    text = f"Angle: {rotation_angle:.2f} deg"
                    
                    # 텍스트 속성 설정
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 2
                    text_color = (0, 255, 255) # 노란색

                    # 텍스트 크기 계산
                    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

                    # 텍스트를 바운딩 박스 하단 중앙에 배치 (x1, y2 기준)
                    # 텍스트가 바운딩 박스보다 약간 아래에 오도록 y2 + padding_for_text 사용
                    text_x = x1 + (x2 - x1 - text_w) // 2
                    text_y = y2 + text_h + 10 # 바운딩 박스 하단에서 10픽셀 아래

                    # 이미지 경계를 벗어나지 않도록 조정
                    text_y = min(text_y, h - 5) # 이미지 하단 경계에서 5픽셀 위
                    text_x = max(0, text_x)
                    text_x = min(text_x, w - text_w)

                    cv2.putText(annotated_frame, text, (text_x, text_y), 
                                font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        cv2.imshow("YOLO + RealSense", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()