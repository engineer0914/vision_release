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
print("YOLO 모델 로드 완료")

# ------------------- 유틸 함수 -------------------

def rs_pointcloud_from_depth(depth_frame, intr, depth_scale):
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth * depth_scale
    x = (u - intr.ppx) * z / intr.fx
    y = (v - intr.ppy) * z / intr.fy
    pts = np.dstack((x, y, z)).reshape(-1, 3)
    return pts, u.reshape(-1), v.reshape(-1)

def make_ground_transform(plane):
    a, b, c, d = plane
    n = np.array([a, b, c])
    n /= np.linalg.norm(n)
    z = np.array([0, 0, 1.0])

    v = np.cross(n, z)
    s = np.linalg.norm(v)
    c_ = np.dot(n, z)

    if s < 1e-6:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c_) / s**2)

    p0 = -d * n
    t = -R @ p0
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def rotation_matrix_to_euler_zyx(R):
    pitch = math.asin(-R[2, 0])
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])

def create_frame(center, R):
    f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=COORD_FRAME_SIZE)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center
    f.transform(T)
    return f

# ------------------- 메인 -------------------

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    align = rs.align(rs.stream.color)

    vis = o3d.visualization.Visualizer()
    vis.create_window("Ground Fixed 3D View", 1280, 720)

    pcd_show = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_show)

    ground_T = None
    locked = False

    print("스트리밍 시작 (q 종료)")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            d = frames.get_depth_frame()
            c = frames.get_color_frame()
            if not d or not c:
                continue

            color = np.asanyarray(c.get_data())
            h, w, _ = color.shape

            yres = model.predict(color, imgsz=640, verbose=False)
            boxes = yres[0].boxes.xyxy.cpu().numpy() if yres[0].boxes else []

            pts, u, v = rs_pointcloud_from_depth(d, intr, depth_scale)
            mask = pts[:, 2] > 0
            pts = pts[mask]
            u, v = u[mask], v[mask]

            pts[:, 1] *= -1  # Open3D 좌표계

            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            down = pcd.voxel_down_sample(VOXEL_DOWNSAMPLE)

            plane, inliers = down.segment_plane(RANSAC_DIST, 3, 500)

            if not locked:
                ground_T = make_ground_transform(plane)
                locked = True
                print(" Ground 기준 좌표계 고정 완료")

            down.transform(ground_T)
            outlier = down.select_by_index(inliers, invert=True)

            pcd_show.points = outlier.points
            vis.update_geometry(pcd_show)

            for box in boxes.astype(int):
                x1, y1, x2, y2 = box
                roi = (u >= x1) & (u < x2) & (v >= y1) & (v < y2)
                if roi.sum() < MIN_POINTS_PER_ROI:
                    continue

                roi_pts = pts[roi]
                roi_pts = (ground_T[:3, :3] @ roi_pts.T).T + ground_T[:3, 3]

                p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(roi_pts))
                obb = o3d.geometry.OrientedBoundingBox.create_from_points(p.points)
                obb.color = (1, 0, 0)

                R = np.asarray(obb.R)
                rpy = rotation_matrix_to_euler_zyx(R)

                vis.add_geometry(obb)
                vis.add_geometry(create_frame(obb.center, R))

                cv2.rectangle(color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color, f"rpy={rpy.round(1)}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            vis.poll_events()
            vis.update_renderer()

            cv2.imshow("2D Pose", color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
