import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from ultralytics import YOLO


# -----------------------------------------------------------
# YOLO 모델 로드
# -----------------------------------------------------------
model = YOLO("../runs/detect/0128_train/weights/best.pt")
print("YOLO 모델 로드 완료!")


def rs_pointcloud_from_frames(depth_frame, intrinsics):
    depth_image = np.asanyarray(depth_frame.get_data())

    h, w = depth_image.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    depth_m = depth_image * intrinsics.depth_scale

    X = (u - intrinsics.ppx) * depth_m / intrinsics.fx
    Y = (v - intrinsics.ppy) * depth_m / intrinsics.fy
    Z = depth_m

    pts = np.dstack((X, Y, Z))
    return pts.reshape(-1, 3), u, v


def main():
    # -----------------------------------------------------------
    # RealSense 설정
    # -----------------------------------------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    intrinsics_data = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    intrinsics_data.depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)

    print("카메라 준비 중...")

    vis = o3d.visualization.Visualizer()
    vis.create_window("3D View", width=1280, height=720)
    geom_added = False

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # -----------------------------------------------------------
            # 1) YOLO 객체 감지
            # -----------------------------------------------------------
            yolo_results = model.predict(color_image, imgsz=640, verbose=False)

            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            names = yolo_results[0].boxes.cls.cpu().numpy()

            # -----------------------------------------------------------
            # 2) 포인트 클라우드 생성
            # -----------------------------------------------------------
            pcd_all, u_map, v_map = rs_pointcloud_from_frames(depth_frame, intrinsics_data)

            valid_idx = pcd_all[:, 2] > 0
            pcd_valid = pcd_all[valid_idx]
            u_valid = u_map.reshape(-1)[valid_idx]
            v_valid = v_map.reshape(-1)[valid_idx]

            # -----------------------------------------------------------
            # 3) 지면 분리 (RANSAC)
            # -----------------------------------------------------------
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd_valid)

            down = pcd_o3d.voxel_down_sample(0.01)

            plane_model, inliers = down.segment_plane(
                distance_threshold=0.01,
                ransac_n=6,
                num_iterations=800
            )

            inlier_cloud = down.select_by_index(inliers)
            outlier_cloud = down.select_by_index(inliers, invert=True)

            # -----------------------------------------------------------
            # 4) 각 ROI별 3D 포인트 추출 + 무게중심 + 3D BBox 계산
            # -----------------------------------------------------------
            bbox_geometries = []

            for idx, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):

                roi_mask = (u_valid >= x1) & (u_valid < x2) & \
                           (v_valid >= y1) & (v_valid < y2)

                roi_pts = pcd_valid[roi_mask]

                if len(roi_pts) < 20:
                    continue

                # ---------- 무게중심 ----------
                centroid = roi_pts.mean(axis=0)

                # ---------- 3D Bounding Box 계산 ----------
                min_pt = roi_pts.min(axis=0)
                max_pt = roi_pts.max(axis=0)

                width = max_pt[0] - min_pt[0]
                height = max_pt[1] - min_pt[1]
                depth = max_pt[2] - min_pt[2]

                # ---------- 영상에 표시 ----------
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.putText(color_image,
                            f"C:({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f})",
                            (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.putText(color_image,
                            f"W:{width:.2f} H:{height:.2f} D:{depth:.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 0), 2)

                cv2.rectangle(color_image, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                # ---------- Open3D OBB 생성 ----------
                obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(roi_pts))
                obb.color = (1, 0, 0)  # red box
                bbox_geometries.append(obb)

            # -----------------------------------------------------------
            # 5) Open3D 시각화 업데이트
            # -----------------------------------------------------------
            if not geom_added:
                vis.add_geometry(outlier_cloud)
                geom_added = True

            vis.update_geometry(outlier_cloud)

            for g in bbox_geometries:
                vis.add_geometry(g)

            vis.poll_events()
            vis.update_renderer()

            # -----------------------------------------------------------
            # 6) OpenCV 출력
            # -----------------------------------------------------------
            cv2.imshow("YOLO + Centroid + 3D Box", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 3D 박스가 계속 쌓이지 않도록 제거
            for g in bbox_geometries:
                vis.remove_geometry(g)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()
        print("종료 완료")


if __name__ == "__main__":
    main()
