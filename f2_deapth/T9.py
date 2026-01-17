import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from ultralytics import YOLO
import math
import time

# ------------------- 사용자 설정 -------------------
YOLO_WEIGHTS = "./runs/detect/train_8n/weights/best.pt"

VOXEL_DOWNSAMPLE = 0.02     # (추천) 0.01 -> 0.02
RANSAC_DIST = 0.02          # (추천) 0.01 -> 0.02
MIN_POINTS_PER_ROI = 150    # (추천) 20 -> 150 (노이즈 ROI 억제)
COORD_FRAME_SIZE = 0.05

# 지면(월드) 좌표계 업데이트 관련
UPDATE_EVERY_N = 5          # N프레임마다 지면 업데이트 시도
INLIER_RATIO_MIN = 0.35     # 지면으로 판단되는 점 비율이 이보다 작으면 업데이트 안 함
EMA_ALPHA_NORMAL = 0.9      # normal 평활(0.8~0.95 추천)
EMA_ALPHA_D = 0.9           # d 평활

# 3D에 무엇을 보여줄지
SHOW_ROI_ONLY = True        # True: ROI 점군만 표시(흔들림 최소), False: ground 제거 outlier 표시
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

def make_ground_transform_from_normal_d(n, d):
    """
    plane: n.x + d = 0 (여기서 n은 단위벡터 가정)
    n을 +Z로 정렬하는 회전 + 평면이 z=0이 되도록 평행이동.
    """
    n = np.asarray(n, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    z = np.array([0, 0, 1.0], dtype=np.float64)

    v = np.cross(n, z)
    s = np.linalg.norm(v)
    c_ = float(np.dot(n, z))

    if s < 1e-6:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]], dtype=np.float64)
        R = np.eye(3) + vx + vx @ vx * ((1 - c_) / (s**2 + 1e-12))

    # 평면 위 한 점 p0 (n·p0 + d = 0) => p0 = -d*n
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

def create_frame(center, R, size=COORD_FRAME_SIZE):
    f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
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

    # (선택) 월드 기준축 하나 고정으로 추가하면 시각적으로 안정감 커짐
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(world_frame, reset_bounding_box=False)

    # 매 프레임 업데이트할 point cloud (한 개만 유지)
    pcd_show = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_show, reset_bounding_box=False)

    # 매 프레임 생성/삭제할 OBB/프레임들
    dyn_geoms = []

    # ground transform 상태
    locked = False
    ground_T = None
    smooth_n = None
    smooth_d = None
    frame_count = 0

    print("스트리밍 시작 (q 종료)")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            dframe = frames.get_depth_frame()
            cframe = frames.get_color_frame()
            if not dframe or not cframe:
                continue

            color = np.asanyarray(cframe.get_data())
            h, w, _ = color.shape

            # YOLO
            yres = model.predict(color, imgsz=640, verbose=False)
            if len(yres) > 0 and hasattr(yres[0], "boxes") and yres[0].boxes is not None:
                boxes = yres[0].boxes.xyxy.cpu().numpy()
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)

            # point cloud
            pts, u, v = rs_pointcloud_from_depth(dframe, intr, depth_scale)
            valid = pts[:, 2] > 0
            pts = pts[valid]
            u, v = u[valid], v[valid]

            # Open3D 좌표계로 변환 (y flip)
            pts[:, 1] *= -1

            if pts.shape[0] < 100:
                cv2.imshow("2D Pose", color)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            down = pcd.voxel_down_sample(VOXEL_DOWNSAMPLE)

            # ---- 지면 추정(RANSAC) ----
            plane, inliers = down.segment_plane(distance_threshold=RANSAC_DIST,
                                                ransac_n=3,
                                                num_iterations=300)

            inlier_ratio = len(inliers) / max(1, len(down.points))
            a, b, c, d = plane
            n = np.array([a, b, c], dtype=np.float64)
            n_norm = np.linalg.norm(n) + 1e-12
            n = n / n_norm
            d = d / n_norm  # 정규화된 n에 맞춰 d도 스케일 보정

            # normal 방향 뒤집힘 방지(프레임마다 부호가 뒤집히면 흔들림)
            if smooth_n is not None and np.dot(n, smooth_n) < 0:
                n = -n
                d = -d

            # ---- ground_T 업데이트 정책 ----
            do_update = (inlier_ratio >= INLIER_RATIO_MIN) and (frame_count % UPDATE_EVERY_N == 0)

            if not locked:
                # 처음에는 신뢰도 충분할 때만 lock
                if inlier_ratio >= INLIER_RATIO_MIN:
                    smooth_n = n.copy()
                    smooth_d = float(d)
                    ground_T = make_ground_transform_from_normal_d(smooth_n, smooth_d)
                    locked = True
                    print(f"[LOCK] Ground 기준 좌표계 고정 완료 (inlier_ratio={inlier_ratio:.2f})")
            else:
                # locked 이후에는 조건부로 EMA 업데이트
                if do_update:
                    smooth_n = EMA_ALPHA_NORMAL * smooth_n + (1 - EMA_ALPHA_NORMAL) * n
                    smooth_n = smooth_n / (np.linalg.norm(smooth_n) + 1e-12)
                    smooth_d = EMA_ALPHA_D * smooth_d + (1 - EMA_ALPHA_D) * float(d)
                    ground_T = make_ground_transform_from_normal_d(smooth_n, smooth_d)

            if not locked:
                # 아직 ground_T가 없다면 그냥 화면만 표시
                cv2.imshow("2D Pose", color)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_count += 1
                continue

            # ---- ground 좌표계로 변환 ----
            down_g = o3d.geometry.PointCloud(down)  # 복사
            down_g.transform(ground_T)

            outlier = down_g.select_by_index(inliers, invert=True)

            # ---- 이전 프레임 OBB/프레임 제거 (누적 방지) ----
            for g in dyn_geoms:
                try:
                    vis.remove_geometry(g, reset_bounding_box=False)
                except:
                    pass
            dyn_geoms = []

            # ---- 3D에 표시할 점군 결정 ----
            roi_points_all = []

            # 박스별 ROI 처리
            for box in boxes.astype(int):
                x1, y1, x2, y2 = box
                x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
                y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                roi = (u >= x1) & (u < x2) & (v >= y1) & (v < y2)
                if roi.sum() < MIN_POINTS_PER_ROI:
                    continue

                roi_pts = pts[roi]  # open3d 좌표계(=y flip) 카메라 좌표
                # ground 좌표계로 변환
                roi_pts_g = (ground_T[:3, :3] @ roi_pts.T).T + ground_T[:3, 3]

                roi_points_all.append(roi_pts_g)

                # OBB 계산
                p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(roi_pts_g))
                obb = o3d.geometry.OrientedBoundingBox.create_from_points(p.points)
                obb.color = (1, 0, 0)

                R = np.asarray(obb.R)
                rpy = rotation_matrix_to_euler_zyx(R)

                # Open3D에 추가 (누적 방지용으로 dyn_geoms에 등록)
                vis.add_geometry(obb, reset_bounding_box=False)
                dyn_geoms.append(obb)

                frame = create_frame(obb.center, R, size=COORD_FRAME_SIZE)
                vis.add_geometry(frame, reset_bounding_box=False)
                dyn_geoms.append(frame)

                # 2D 표시
                cv2.rectangle(color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color, f"rpy={np.round(rpy,1)}",
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # point cloud 표시 업데이트
            if SHOW_ROI_ONLY:
                if len(roi_points_all) > 0:
                    show_pts = np.vstack(roi_points_all)
                else:
                    show_pts = np.zeros((0, 3), dtype=np.float64)  # ROI 없으면 빈 점군
            else:
                show_pts = np.asarray(outlier.points)

            pcd_show.points = o3d.utility.Vector3dVector(show_pts)
            vis.update_geometry(pcd_show)

            vis.poll_events()
            vis.update_renderer()

            cv2.imshow("2D Pose", color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            time.sleep(0.001)

    finally:
        pipeline.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
