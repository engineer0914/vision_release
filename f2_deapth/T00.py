# T8_ground_yolo_pose.py (axis-fixed) + stage windows
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from ultralytics import YOLO
import math
import time

# ------------------- 사용자 설정 -------------------
YOLO_WEIGHTS = "./runs/detect/train_8n/weights/best.pt"
VOXEL_DOWNSAMPLE = 0.01
RANSAC_DIST = 0.01
MIN_POINTS_PER_ROI = 20
COORD_FRAME_SIZE = 0.05
# --------------------------------------------------

model = YOLO(YOLO_WEIGHTS)
print("YOLO 모델 로드 완료:", YOLO_WEIGHTS)

def rs_pointcloud_from_depth(depth_frame, depth_intrinsics, depth_scale):
    depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    h, w = depth_image.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    depth_m = depth_image * depth_scale

    X = (u - depth_intrinsics.ppx) * depth_m / depth_intrinsics.fx
    Y = (v - depth_intrinsics.ppy) * depth_m / depth_intrinsics.fy
    Z = depth_m

    pts = np.dstack((X, Y, Z)).reshape(-1, 3)
    return pts, u, v, depth_image

def rotation_matrix_to_euler_zyx(R):
    r00,r01,r02 = R[0,0],R[0,1],R[0,2]
    r10,r11,r12 = R[1,0],R[1,1],R[1,2]
    r20,r21,r22 = R[2,0],R[2,1],R[2,2]
    if abs(r20) < 0.9999:
        pitch = math.asin(-r20)
        cos_p = math.cos(pitch)
        roll = math.atan2(r21 / cos_p, r22 / cos_p)
        yaw  = math.atan2(r10 / cos_p, r00 / cos_p)
    else:
        pitch = math.asin(-r20)
        yaw = 0.0
        if r20 <= -0.9999:
            roll = math.atan2(-r01, -r02)
        else:
            roll = math.atan2(r01, r02)
    return roll, pitch, yaw

def deg(rad): return rad * 180.0 / math.pi

def create_coordinate_frame_at(center, R, size=0.05):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = center
    frame.transform(T)
    return frame

def main():
    # ---------------- RealSense 설정 ----------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    depth_stream_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_intr = depth_stream_profile.get_intrinsics()

    align = rs.align(rs.stream.color)

    # ---------------- Open3D 시각화 ----------------
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("3D Stages (1~6)", width=1280, height=720)

    # stage:
    # 1 raw(camera) / 2 yflip / 3 downsample / 4 ground / 5 non-ground+obb / 6 roi-only
    state = {"stage": 5, "sel_idx": 0}  # sel_idx: ROI 선택 인덱스

    def set_stage(s):
        def _cb(v):
            state["stage"] = s
            print(f"[3D] stage = {s}")
            return False
        return _cb

    # 키 바인딩
    vis.register_key_callback(ord('1'), set_stage(1))
    vis.register_key_callback(ord('2'), set_stage(2))
    vis.register_key_callback(ord('3'), set_stage(3))
    vis.register_key_callback(ord('4'), set_stage(4))
    vis.register_key_callback(ord('5'), set_stage(5))
    vis.register_key_callback(ord('6'), set_stage(6))

    # ROI 선택 바꾸기(여러 객체 잡힐 때 확인용)
    def next_roi(v):
        state["sel_idx"] += 1
        print(f"[ROI] select index = {state['sel_idx']}")
        return False
    def prev_roi(v):
        state["sel_idx"] = max(0, state["sel_idx"] - 1)
        print(f"[ROI] select index = {state['sel_idx']}")
        return False

    vis.register_key_callback(ord(']'), next_roi)
    vis.register_key_callback(ord('['), prev_roi)

    print("준비 완료.")
    print("Open3D 키:")
    print("  1: Raw PointCloud(카메라 좌표계)")
    print("  2: Y-flip PointCloud(Open3D 좌표계)")
    print("  3: Downsample PointCloud")
    print("  4: Ground only(inliers)")
    print("  5: Non-ground(outliers)+OBB+Frame")
    print("  6: ROI 3D points only(선택 객체)")
    print("  [ / ]: ROI 선택 이전/다음 (객체 여러 개일 때)")
    print("OpenCV 창:")
    print("  Color, Depth, ROI Mask, ROI Depth")
    print("q: 종료")

    added_geoms = []
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            h, w, _ = color_image.shape

            # ------------- (A) YOLO -------------
            yres = model.predict(color_image, imgsz=640, verbose=False)
            if len(yres) > 0 and hasattr(yres[0], "boxes"):
                boxes = yres[0].boxes.xyxy.cpu().numpy()
                scores = yres[0].boxes.conf.cpu().numpy()
                classes = yres[0].boxes.cls.cpu().numpy()
            else:
                boxes = np.zeros((0,4))
                scores = np.zeros((0,))
                classes = np.zeros((0,))

            # ------------- (B) Depth -> PointCloud (카메라 좌표계) -------------
            pts_all, u_map, v_map, depth_u16 = rs_pointcloud_from_depth(depth_frame, depth_intr, depth_scale)
            u_flat = u_map.reshape(-1)
            v_flat = v_map.reshape(-1)

            valid_mask = pts_all[:, 2] > 0
            pts_valid = pts_all[valid_mask]
            u_valid = u_flat[valid_mask]
            v_valid = v_flat[valid_mask]

            # (시각화용) Open3D 좌표계 맞추기: Y축 flip
            pts_valid_transformed = pts_valid.copy()
            pts_valid_transformed[:, 1] *= -1.0

            if len(pts_valid) < 50:
                # OpenCV만 띄우고 다음 루프
                cv2.imshow("01_Color(YOLO)", color_image)
                depth_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03),
                    cv2.COLORMAP_JET
                )
                cv2.imshow("02_Depth(colormap)", depth_vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ------------- (C) Open3D cloud 만들기 -------------
            pcd_raw = o3d.geometry.PointCloud()
            pcd_raw.points = o3d.utility.Vector3dVector(pts_valid)  # 카메라 좌표계 raw

            pcd_yflip = o3d.geometry.PointCloud()
            pcd_yflip.points = o3d.utility.Vector3dVector(pts_valid_transformed)  # Open3D용

            # ------------- (D) Downsample + Ground plane 분리(시각화용) -------------
            down = pcd_yflip.voxel_down_sample(VOXEL_DOWNSAMPLE)
            plane_model, inliers = down.segment_plane(distance_threshold=RANSAC_DIST,
                                                      ransac_n=6,
                                                      num_iterations=800)
            inlier_cloud = down.select_by_index(inliers)                 # ground
            outlier_cloud = down.select_by_index(inliers, invert=True)   # non-ground

            # ------------- (E) ROI 마스크 / ROI 포인트 / OBB 계산 -------------
            bbox_geoms = []
            frame_geoms = []
            roi_only_cloud = None

            # ROI 선택을 위해 현재 box 리스트에서 sel_idx 적용
            # (객체 여러 개면 [ ]로 바꿔가며 확인)
            sel_idx = state["sel_idx"]
            sel_idx = min(sel_idx, max(0, len(boxes)-1))
            state["sel_idx"] = sel_idx

            roi_mask_img = np.zeros((h, w), dtype=np.uint8)
            roi_depth_only = np.zeros((h, w), dtype=np.uint16)

            for i, box in enumerate(boxes.astype(int)):
                x1, y1, x2, y2 = box
                x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
                y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))

                # 2D ROI에 해당하는 3D 점 마스킹(정렬된 depth 기반)
                mask_roi = (u_valid >= x1) & (u_valid < x2) & (v_valid >= y1) & (v_valid < y2)
                roi_pts_camera = pts_valid[mask_roi]
                roi_pts_vis = pts_valid_transformed[mask_roi]

                if roi_pts_camera.shape[0] < MIN_POINTS_PER_ROI:
                    continue

                # 선택된 ROI에 대해서만 “ROI 창”에 표시하기 위한 마스크 생성
                if i == sel_idx:
                    roi_mask_img[y1:y2, x1:x2] = 255
                    # depth 원본에서 ROI만 남겨보기(정렬된 depth)
                    depth_img_u16 = np.asanyarray(depth_frame.get_data())
                    roi_depth_only[y1:y2, x1:x2] = depth_img_u16[y1:y2, x1:x2]

                    # ROI-only 3D point cloud(시각화 좌표계)도 만들기
                    roi_only_cloud = o3d.geometry.PointCloud()
                    roi_only_cloud.points = o3d.utility.Vector3dVector(roi_pts_vis)

                # OBB 생성(시각화 좌표계)
                pcd_roi = o3d.geometry.PointCloud()
                pcd_roi.points = o3d.utility.Vector3dVector(roi_pts_vis)
                obb = o3d.geometry.OrientedBoundingBox.create_from_points(pcd_roi.points)
                obb.color = (1.0, 0.0, 0.0)

                center_vis = obb.center
                extent = obb.extent
                R_vis = np.asarray(obb.R)

                roll, pitch, yaw = rotation_matrix_to_euler_zyx(R_vis)
                roll_d, pitch_d, yaw_d = deg(roll), deg(pitch), deg(yaw)

                # 2D 표시
                cv2.rectangle(color_image, (x1, y1), (x2, y2),
                              (0, 255, 255) if i == sel_idx else (0, 255, 0), 2)

                center_cam_from_vis = center_vis.copy()
                center_cam_from_vis[1] *= -1.0

                cv2.putText(color_image, f"[{i}] W:{extent[0]:.2f} H:{extent[1]:.2f} D:{extent[2]:.2f}m",
                            (x1, max(0,y1-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,255,200), 2)
                cv2.putText(color_image, f"r:{roll_d:.1f} p:{pitch_d:.1f} y:{yaw_d:.1f}",
                            (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,255), 2)

                bbox_geoms.append(obb)
                frame_geoms.append(create_coordinate_frame_at(center_vis, R_vis, size=COORD_FRAME_SIZE))

            # ------------- OpenCV 단계별 창 표시(2D) -------------
            # 1) Color
            cv2.imshow("01_Color(YOLO)", color_image)

            # 2) Depth colormap
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.imshow("02_Depth(colormap)", depth_vis)

            # 3) ROI Mask
            roi_mask_bgr = cv2.cvtColor(roi_mask_img, cv2.COLOR_GRAY2BGR)
            cv2.imshow("03_ROI_Mask(selected)", roi_mask_bgr)

            # 4) ROI Depth only
            roi_depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(roi_depth_only, alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.imshow("04_ROI_Depth(selected)", roi_depth_vis)

            # ------------- Open3D 단계별 표시(3D) -------------
            # 이전 geometry 제거
            for g in added_geoms:
                try: vis.remove_geometry(g)
                except: pass
            added_geoms = []

            stage = state["stage"]

            # stage별로 다른 지오메트리 추가
            if stage == 1:
                vis.add_geometry(pcd_raw); added_geoms.append(pcd_raw)

            elif stage == 2:
                vis.add_geometry(pcd_yflip); added_geoms.append(pcd_yflip)

            elif stage == 3:
                vis.add_geometry(down); added_geoms.append(down)

            elif stage == 4:
                vis.add_geometry(inlier_cloud); added_geoms.append(inlier_cloud)

            elif stage == 5:
                vis.add_geometry(outlier_cloud); added_geoms.append(outlier_cloud)
                for g in bbox_geoms:
                    vis.add_geometry(g); added_geoms.append(g)
                for f in frame_geoms:
                    vis.add_geometry(f); added_geoms.append(f)

            elif stage == 6:
                # 선택 ROI 3D 점만 보기(없으면 아무것도 안 나옴)
                if roi_only_cloud is not None:
                    vis.add_geometry(roi_only_cloud); added_geoms.append(roi_only_cloud)
                for g in bbox_geoms:
                    vis.add_geometry(g); added_geoms.append(g)
                for f in frame_geoms:
                    vis.add_geometry(f); added_geoms.append(f)

            vis.poll_events()
            vis.update_renderer()

            # 종료키
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

            time.sleep(0.001)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()
        print("종료 완료.")

if __name__ == "__main__":
    main()
