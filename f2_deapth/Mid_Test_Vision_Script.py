# T8_ground_yolo_pose_fixed_v2.py
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from ultralytics import YOLO
import math
import time
import sys

# ------------------- 사용자 설정 -------------------
YOLO_WEIGHTS = "../../runs/detect/train_8n/weights/best.pt"
VOXEL_DOWNSAMPLE = 0.01        
RANSAC_DIST = 0.01             
MIN_POINTS_PER_ROI = 20        
COORD_FRAME_SIZE = 0.05        
# --------------------------------------------------

model = YOLO(YOLO_WEIGHTS)
# YOLO verbose 끄기 (터미널 오염 방지)
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def rs_pointcloud_from_depth(depth_frame, depth_intrinsics, depth_scale):
    depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    h, w = depth_image.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    depth_m = depth_image * depth_scale
    X = (u - depth_intrinsics.ppx) * depth_m / depth_intrinsics.fx
    Y = (v - depth_intrinsics.ppy) * depth_m / depth_intrinsics.fy
    Z = depth_m
    return np.dstack((X, Y, Z)).reshape(-1, 3), u, v

def rotation_matrix_to_euler_zyx(R):
    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]
    if abs(r20) < 0.9999:
        pitch = math.asin(-r20)
        cos_p = math.cos(pitch)
        roll = math.atan2(r21 / cos_p, r22 / cos_p)
        yaw = math.atan2(r10 / cos_p, r00 / cos_p)
    else:
        pitch = math.asin(-r20)
        yaw = 0.0
        roll = math.atan2(-r01, -r02) if r20 <= -0.9999 else math.atan2(r01, r02)
    return roll, pitch, yaw

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    align = rs.align(rs.stream.color)

    vis = o3d.visualization.Visualizer()
    vis.create_window("3D View", width=1280, height=720)
    added_geoms = [] 

    print("\n=== 시스템 준비 완료 ===")
    print("Spacebar : Hold to Calculate 6D Pose")
    print("Q        : Quit\n")

    key_input = -1
    last_mode = "init"  # 터미널 출력을 제어하기 위한 상태 변수

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame: continue

            color_image = np.asanyarray(color_frame.get_data())
            h, w, _ = color_image.shape

            # YOLO (항상 실행)
            yres = model.predict(color_image, imgsz=640, verbose=False)
            boxes = yres[0].boxes.xyxy.cpu().numpy() if len(yres) > 0 and hasattr(yres[0], "boxes") else np.zeros((0, 4))
            
            # 스페이스바(32) 눌림 확인
            is_calculating = (key_input == 32)

            if is_calculating:
                # ---------------- [계산 모드] ----------------
                pts_all, u_map, v_map = rs_pointcloud_from_depth(depth_frame, depth_intr, depth_scale)
                
                # 좌표 변환 및 유효 포인트 필터링
                valid_mask = pts_all[:, 2] > 0
                pts_valid = pts_all[valid_mask]
                u_valid = u_map.reshape(-1)[valid_mask]
                v_valid = v_map.reshape(-1)[valid_mask]
                
                pts_vis = pts_valid.copy()
                pts_vis[:, 1] *= -1.0 # Vis용 Y반전

                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(pts_vis)

                if len(pts_vis) < 50:
                    cv2.imshow("Cam", color_image)
                    key_input = cv2.waitKey(1)
                    if key_input & 0xFF == ord('q'): break
                    continue

                # RANSAC
                down = pcd_o3d.voxel_down_sample(VOXEL_DOWNSAMPLE)
                if len(down.points) >= 4:
                    _, inliers = down.segment_plane(RANSAC_DIST, 3, 500)
                    outlier_cloud = down.select_by_index(inliers, invert=True) if len(inliers) > 0 else down
                else:
                    outlier_cloud = down

                bbox_geoms = []
                frame_geoms = []
                terminal_str = ""

                for i, box in enumerate(boxes.astype(int)):
                    x1, y1, x2, y2 = box
                    x1, x2 = np.clip([x1, x2], 0, w-1)
                    y1, y2 = np.clip([y1, y2], 0, h-1)
                    
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    mask_roi = (u_valid >= x1) & (u_valid < x2) & (v_valid >= y1) & (v_valid < y2)
                    roi_pts_vis = pts_vis[mask_roi]

                    if roi_pts_vis.shape[0] < MIN_POINTS_PER_ROI: continue

                    pcd_roi = o3d.geometry.PointCloud()
                    pcd_roi.points = o3d.utility.Vector3dVector(roi_pts_vis)

                    try:
                        obb = o3d.geometry.OrientedBoundingBox.create_from_points(pcd_roi.points)
                    except RuntimeError: continue

                    obb.color = (1, 0, 0)
                    center_vis = obb.center
                    R_vis = np.asarray(obb.R)
                    
                    # Vis -> Camera 좌표 변환 (Y 다시 반전)
                    cx, cy, cz = center_vis[0], -center_vis[1], center_vis[2]
                    roll, pitch, yaw = rotation_matrix_to_euler_zyx(R_vis)
                    r_d, p_d, y_d = math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

                    cv2.putText(color_image, f"XYZ:{cx:.2f},{cy:.2f},{cz:.2f}", (x1, y1-20), 0, 0.5, (0,255,255), 2)
                    
                    # 터미널 메시지 누적
                    terminal_str += f"[Obj{i}] {cx:.2f},{cy:.2f},{cz:.2f} | RPY:{r_d:.0f},{p_d:.0f},{y_d:.0f}  "
                    
                    bbox_geoms.append(obb)
                    frame_geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=COORD_FRAME_SIZE).transform(
                        np.vstack((np.hstack((R_vis, center_vis.reshape(3,1))), [0,0,0,1]))
                    ))

                # --- 터미널 업데이트 (계산 중엔 계속 갱신) ---
                if not terminal_str: terminal_str = "Detecting..."
                sys.stdout.write(f"\r{terminal_str:<100}") # 100칸 공백으로 뒤쪽 잔상 지움
                sys.stdout.flush()
                last_mode = "calc"

                # 3D 뷰 업데이트
                for g in added_geoms: vis.remove_geometry(g)
                added_geoms = [outlier_cloud] + bbox_geoms + frame_geoms
                for g in added_geoms: vis.add_geometry(g)
                vis.poll_events()
                vis.update_renderer()

            else:
                # ---------------- [대기 모드] ----------------
                # 박스만 그리기
                for box in boxes.astype(int):
                    cv2.rectangle(color_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                # ★ 중요: 대기 모드 메시지는 '상태가 변할 때 한 번만' 출력 (스크롤 방지)
                if last_mode != "standby":
                    sys.stdout.write(f"\r[Standby] Press Spacebar...{' '*60}")
                    sys.stdout.flush()
                    last_mode = "standby"
                
                vis.poll_events()
                vis.update_renderer()

            cv2.imshow("Cam", color_image)
            key_input = cv2.waitKey(1)
            if key_input & 0xFF == ord('q'): break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()
        print("\nExit.")

if __name__ == "__main__":
    main()

