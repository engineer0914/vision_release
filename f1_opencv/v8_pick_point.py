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
                
                rotation_angle = None
                global_moment_center_x = None
                global_moment_center_y = None
                
                if roi_color.shape[0] > 0 and roi_color.shape[1] > 0:
                    gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray_roi, canny_threshold1, canny_threshold2)
                    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                    
                    contours, hierarchy = cv2.findContours(dilated_edges, 
                                                           cv2.RETR_EXTERNAL, 
                                                           cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)

                        if len(largest_contour) >= min_contour_length_for_ellipse:
                            ellipse = cv2.fitEllipse(largest_contour)
                            rotation_angle = ellipse[2]
                            cv2.ellipse(roi_color, ellipse, (0, 0, 255), 2)
                        else:
                            cv2.drawContours(roi_color, [largest_contour], -1, (0, 255, 0), 2)

                        M = cv2.moments(largest_contour)

                        if M['m00'] != 0:
                            roi_moment_center_x = int(M['m10'] / M['m00'])
                            roi_moment_center_y = int(M['m01'] / M['m00'])
                            
                            global_moment_center_x = roi_moment_center_x + x1_pad
                            global_moment_center_y = roi_moment_center_y + y1_pad
                            
                            cv2.circle(roi_color, (roi_moment_center_x, roi_moment_center_y), 5, (255, 255, 255), -1)

                    cv2.imshow(f"Ellipse and Moment Center on ROI: {class_name}", roi_color)

                # 메인 annotated_frame에 모멘트 중심점만 표시 (초록색 원)
                if global_moment_center_x is not None and global_moment_center_y is not None:
                    cv2.circle(annotated_frame, (global_moment_center_x, global_moment_center_y), 5, (0, 255, 0), -1)

                # 메인 annotated_frame에 회전 각도 텍스트만 표시
                if rotation_angle is not None:
                    text = f"Angle: {rotation_angle:.2f} deg"
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 2
                    text_color = (0, 255, 255) # 노란색

                    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

                    text_x = x1 + (x2 - x1 - text_w) // 2
                    text_y = y2 + text_h + 10

                    text_y = min(text_y, h - 5)
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