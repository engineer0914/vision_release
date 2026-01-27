import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 1. 파이프라인 설정
    pipeline = rs.pipeline()
    config = rs.config()

    # 2. 스트림 설정 (가장 안전한 640x480 30fps로 시도)
    # D435if가 확실히 지원하는 모드입니다.
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 3. 카메라 시작
    print("카메라 연결 시도 중...")
    try:
        pipeline.start(config)
        print("✅ 카메라 연결 성공! (Ctrl+C로 종료)")
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return

    try:
        while True:
            # 4. 프레임 받기
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 5. 이미지 변환
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 6. 화면 출력
            # 깊이 이미지는 보기 좋게 컬러맵 적용
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # 두 이미지를 가로로 붙이기
            images = np.hstack((color_image, depth_colormap))

            cv2.imshow('RealSense Test', images)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
