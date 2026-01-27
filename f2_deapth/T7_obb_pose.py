# T8_ground_yolo_pose.py (axis-fixed)
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from ultralytics import YOLO
import math
import time

# ------------------- 사용자 설정 -------------------
YOLO_WEIGHTS = "../runs/detect/0128_train/weights/best.pt"
VOXEL_DOWNSAMPLE = 0.01        # RANSAC 전 다운샘플 크기 (m)
RANSAC_DIST = 0.01             # 평면(ground) distance threshold (m)
MIN_POINTS_PER_ROI = 20        # ROI에서 최소 포인트 수 (무시할 임계값)
COORD_FRAME_SIZE = 0.05        # Open3D 좌표축 크기 (m)
# --------------------------------------------------

# YOLO 로드
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
    return pts, u, v


def rotation_matrix_to_euler_zyx(R):
    """
    Rotation matrix R -> Euler angles in ZYX order (yaw, pitch, roll)
    Returns (roll, pitch, yaw) in radians.
    """
    r00 = R[0, 0]; r01 = R[0, 1]; r02 = R[0, 2]
    r10 = R[1, 0]; r11 = R[1, 1]; r12 = R[1, 2]
    r20 = R[2, 0]; r21 = R[2, 1]; r22 = R[2, 2]

    if abs(r20) < 0.9999:
        pitch = math.asin(-r20)
        cos_p = math.cos(pitch)
        roll = math.atan2(r21 / cos_p, r22 / cos_p)
        yaw = math.atan2(r10 / cos_p, r00 / cos_p)
    else:
        pitch = math.asin(-r20)
        yaw = 0.0
        if r20 <= -0.9999:
            roll = math.atan2(-r01, -r02)
        else:
            roll = math.atan2(r01, r02)

    return roll, pitch, yaw  # radians


def deg(rad):
    return rad * 180.0 / math.pi


def create_coordinate_frame_at(center, R, size=0.05):
    """
    Create an Open3D coordinate frame located at center with rotation R (3x3).
    Returns a TriangleMesh (frame) that has been transformed.
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center
    frame.transform(T)
    return frame


def main():
    # RealSense 설정
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

    # Open3D 시각화(비동기) 설정
    vis = o3d.visualization.Visualizer()
    vis.create_window("3D View (ground removed + OBB+Pose)", width=1280, height=720)
    added_geoms = []  # keep track to remove each loop

    print("카메라 및 시각화 준비 완료. 스트리밍 시작 (q: 종료)")

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

            # 1) YOLO detection on color image
            yres = model.predict(color_image, imgsz=640, verbose=False)
            boxes = yres[0].boxes.xyxy.cpu().numpy() if len(yres) > 0 and hasattr(yres[0], "boxes") else np.zeros((0, 4))
            scores = yres[0].boxes.conf.cpu().numpy() if len(yres) > 0 and hasattr(yres[0], "boxes") else np.zeros((0,))
            classes = yres[0].boxes.cls.cpu().numpy() if len(yres) > 0 and hasattr(yres[0], "boxes") else np.zeros((0,))

            # 2) Depth -> point cloud (camera coords) + u,v maps
            pts_all, u_map, v_map = rs_pointcloud_from_depth(
                depth_frame, depth_intr, depth_scale
            )
            u_flat = u_map.reshape(-1)
            v_flat = v_map.reshape(-1)

            # remove invalid depth
            valid_mask = pts_all[:, 2] > 0
            pts_valid = pts_all[valid_mask]
            u_valid = u_flat[valid_mask]
            v_valid = v_flat[valid_mask]

            # ------------ HERE: transform for visualization ------------
            # Create a transformed copy for Open3D visualization (flip Y)
            # Camera coords: +Y goes down (image row increases downwards)
            # Open3D: +Y goes up -> flip sign
            pts_valid_transformed = pts_valid.copy()
            pts_valid_transformed[:, 1] *= -1.0

            # 3) Build Open3D pointcloud (USE transformed points) and remove ground plane via RANSAC
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pts_valid_transformed)

            if len(pts_valid_transformed) < 50:
                # not enough points; just show color & continue
                cv2.imshow("Pose (OBB) Output", color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            down = pcd_o3d.voxel_down_sample(VOXEL_DOWNSAMPLE)

            plane_model, inliers = down.segment_plane(distance_threshold=RANSAC_DIST,
                                                      ransac_n=6,
                                                      num_iterations=800)
            inlier_cloud = down.select_by_index(inliers)            # ground (transformed coords)
            outlier_cloud = down.select_by_index(inliers, invert=True)  # non-ground (transformed coords)

            # For ROI selection, we use original (camera) pts_valid for masking, but use transformed pts for OBB/vis
            bbox_geoms = []   # store OBBs (transformed coords)
            frame_geoms = []  # store frames for orientation visualization (transformed coords)

            for i, box in enumerate(boxes.astype(int)):
                x1, y1, x2, y2 = box
                # clip
                x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
                y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))

                # mask points lying inside this 2D ROI (using the full-resolution pts_valid & u_valid/v_valid)
                mask_roi = (u_valid >= x1) & (u_valid < x2) & (v_valid >= y1) & (v_valid < y2)
                roi_pts_camera = pts_valid[mask_roi]              # camera coords (for display / raw centroid)
                roi_pts_vis = pts_valid_transformed[mask_roi]     # transformed coords (for OBB / vis)

                if roi_pts_camera.shape[0] < MIN_POINTS_PER_ROI:
                    # not enough points: skip
                    continue

                # compute OBB from ROI points (use transformed points)
                pcd_roi = o3d.geometry.PointCloud()
                pcd_roi.points = o3d.utility.Vector3dVector(roi_pts_vis)
                obb = o3d.geometry.OrientedBoundingBox.create_from_points(pcd_roi.points)
                obb.color = (1.0, 0.0, 0.0)  # red box (transformed coords)

                # OBB properties in transformed (viz) coords
                center_vis = obb.center          # (x,y,z) in transformed coords
                extent = obb.extent              # (W, H, D)
                R_vis = np.asarray(obb.R)        # 3x3 rotation matrix (transformed coords)

                # Euler angles (roll, pitch, yaw) from R (ZYX order)
                roll, pitch, yaw = rotation_matrix_to_euler_zyx(R_vis)
                roll_d, pitch_d, yaw_d = deg(roll), deg(pitch), deg(yaw)

                # Also compute centroid from raw roi_pts (camera coords) for display
                centroid_camera = roi_pts_camera.mean(axis=0)

                # Visual annotations on color image (show camera-coords centroid & OBB size & pose)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(color_image, f"C_cam: {centroid_camera[0]:.2f},{centroid_camera[1]:.2f},{centroid_camera[2]:.2f} m",
                #             (x1, y1 - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
                # also show center in visualization coords converted back to camera coords for readability:
                # center_camera_from_vis = center_vis copy with y flipped back
                center_camera_from_vis = center_vis.copy()
                center_camera_from_vis[1] *= -1.0
                cv2.putText(color_image, f"C=>cam: {center_camera_from_vis[0]:.2f},{center_camera_from_vis[1]:.2f},{center_camera_from_vis[2]:.2f} m",
                            (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 200), 2)
                cv2.putText(color_image, f"W:{extent[0]:.2f} H:{extent[1]:.2f} D:{extent[2]:.2f} m",
                            (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 2)
                cv2.putText(color_image, f"r:{roll_d:.1f} p:{pitch_d:.1f} y:{yaw_d:.1f} deg",
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 2)

                # console print (both camera centroid and vis center + orientation)
                # print(f"[Obj {i}] centroid_cam={centroid_camera}, center_vis={center_vis}, extent={extent}, rpy_deg=({roll_d:.2f},{pitch_d:.2f},{yaw_d:.2f})")
                print(f"[Obj {i}] C=>cam:=x_off({center_camera_from_vis[0]:.2f}),y_off({center_camera_from_vis[1]:.2f}),z_off({center_camera_from_vis[2]:.2f}), rpy_deg=({roll_d:.2f},{pitch_d:.2f},{yaw_d:.2f})")

                # Add OBB to list for Open3D visualization (transformed coords)
                bbox_geoms.append(obb)

                # Create and place a small coordinate frame at obb.center with rotation R_vis (transformed coords)
                frame = create_coordinate_frame_at(center_vis, R_vis, size=COORD_FRAME_SIZE)
                frame_geoms.append(frame)

            # ---------------- Open3D visualization update ----------------
            # Clear previous added geometries to prevent accumulation
            for g in added_geoms:
                try:
                    vis.remove_geometry(g)
                except:
                    pass
            added_geoms = []

            # add non-ground points for context (transformed coords)
            vis.add_geometry(outlier_cloud)
            added_geoms.append(outlier_cloud)

            # add each obb and frame (transformed coords)
            for g in bbox_geoms:
                vis.add_geometry(g)
                added_geoms.append(g)
            for f in frame_geoms:
                vis.add_geometry(f)
                added_geoms.append(f)

            vis.poll_events()
            vis.update_renderer()

            # show color image with overlays
            cv2.imshow("Pose (OBB) Output", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # small sleep to yield CPU (optional)
            time.sleep(0.001)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()
        print("종료 완료.")


if __name__ == "__main__":
    main()
