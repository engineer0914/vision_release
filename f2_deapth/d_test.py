#!pip install pyrealsense2
#!pip install opencv-python
import pyrealsense2 as rs
import cv2
import numpy as np


class realsense_camera:
    is_opened = False;
    config = None;
    intr = None;

    def __init__(self, height = 480, width= 640, fps =30, use_color=True,use_depth = True):
        self.height = height;
        self.width = width;
        self.fps = fps;
        self.use_depth = use_depth;
        self.use_color = use_color;

        # depth and color 설정 변수 생성
        pipeline = rs.pipeline()
        config = rs.config()

        # 디바이스 변수 얻기
        pipeline_wrapper = rs.pipeline_wrapper(pipeline);
        self.config = config;
        self.pipeline = pipeline;
        self.pipeline_wrapper = pipeline_wrapper;

        # 디바이스 연결 체크
        if(self.can_connect()):        
            pipeline_profile = config.resolve(pipeline_wrapper);
            device = pipeline_profile.get_device();

            # 디바이스 내 color 또는 depth 모듈 체크
            found_rgb = False
            found_depth = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'Stereo Module':
                    found_depth = True;
                elif s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True

            # color 또는 depth 모듈이 없다면 비활성화
            if not found_rgb:
                use_color = False; 
            if not found_depth:
                use_depth = False;    
       
            
            # 모듈이 있다면 
            if(self.use_depth):
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            if(self.use_color):
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps);
            
            if(self.can_connect() and (self.use_depth or self.use_color)):
                self.is_opened = True;
                pipeline.start(config);
                if(self.use_color):
                    self.intr = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics();


    def get_intrinsics(self):
        return self.intr;

    def set(self,num,value):                      
        if(num == cv2.CAP_PROP_FRAME_HEIGHT):
            self.height= value;
        elif(num == cv2.CAP_PROP_FRAME_WIDTH):
            self.width= value;
        elif(num == cv2.CAP_PROP_FPS):
            self.fps= value;
        if(self.is_opened):
            if(self.use_depth):
                self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps);
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps);
            if(self.can_connect()):
                self.pipeline.stop();
                self.pipeline.start(self.config);
                self.intr = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics();
    
    def get(self,num):                      
        if(num == cv2.CAP_PROP_FRAME_HEIGHT):
            return self.height;
        elif(num == cv2.CAP_PROP_FRAME_WIDTH):
            return  self.width;
        elif(num == cv2.CAP_PROP_FPS):
            return  self.fps;
    
    def can_connect(self):
        # config의 유효성 및 카메라의 연결여부 체크용도
        return self.config.can_resolve(self.pipeline_wrapper);
    
    def isOpened(self):
        return self.is_opened;
    
    def release(self):
        if(not self.pipeline is None):
            if(self.isOpened()):
                if(self.can_connect()):
                    self.pipeline.stop();
    
    def read_color_depth(self):
        try:
            #파이프 라인으로부터 프레임 얻기 (100ms안에)
            frames = self.pipeline.wait_for_frames(100)
            color_frame, depth_frame = None, None;
            color_image, depth_image = None, None;

            #프레임에서 color와 depth 얻기
            if(self.use_color):
                color_frame = frames.get_color_frame()
            if(self.use_depth):
                depth_frame = frames.get_depth_frame()
                if(not depth_frame):
                    return False,[None, None]
            
            #numpy형으로 변환
            if(not color_frame is None):
                color_image = np.asanyarray(color_frame.get_data())
            if(not depth_frame is None):
                depth_image = np.asanyarray(depth_frame.get_data())
            
            #데이터 반환
            return True if((not color_image is None) or (not depth_image is None)) else False,[color_image, depth_image];
    
        except:
            return False,[None, None]
        
    def read(self):
        try:
            #파이프 라인으로부터 프레임 얻기 (100ms안에)
            frames = self.pipeline.wait_for_frames(100)
            if(self.use_color):
                #프레임에서 color만 얻기
                color_frame = frames.get_color_frame();
                if (not color_frame):
                    return False, None
                #numpy형으로 변환
                color_image = np.asanyarray(color_frame.get_data());
            else:
                color_image = np.zeros((self.height,self.width,3),dtype=np.uint8);
            #데이터 반환
            return True, color_image;
        except:
            return False,None

if __name__ == "__main__":

    cam = realsense_camera(use_color=True,use_depth=True);
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,1280);
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,720);
    cam.set(cv2.CAP_PROP_FPS,30);

    if(cam.isOpened()):
        while True:
            ret, frame = cam.read();
            if(ret):
                cv2.imshow("frame",frame);
            key=cv2.waitKey(1);
            if(key&0xff==ord('q')):
                break
    
        while True:
            ret, frame = cam.read_color_depth();
            if(ret):
                color, depth = frame;
                if(not color is None):
                    cv2.imshow("color",color);
                if(not depth is None):
                    cv2.imshow("depth",depth);
                print(cam.get_intrinsics())
            key=cv2.waitKey(1);
            if(key&0xff==ord('q')):
                break
    
    cam.release()