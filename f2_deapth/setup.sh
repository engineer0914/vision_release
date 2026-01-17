
# 가상환경 구축
#conda env remove -n rbcpc
conda create -n rbcpc python=3.9 -y

# 활성화
conda activate rbcpc

# 패키지 한번에 설치 - 잘 안됨
#pip install pyrealsense2 numpy open3d opencv-python ultralytics

# 따로 따로
pip install pyrealsense2 && \
pip install numpy && \
pip install open3d && \
pip install opencv-python && \
pip install ultralytics && \

: << "END"
# cuda 확인하기
nvidia-smi

# 그래픽 카드에 맞게 pytorch 설치

# 아래를 돌려서 밑 처럼 나오면 됨
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

예)
(rbcpc) recl3090@recl3090-Precision-3680:~/aavision/yolov11/f2_deapth$ python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
PyTorch: 2.8.0+cu128
CUDA Available: True
Device Name: NVIDIA GeForce RTX 3090
END


# --------------------------------------------------------

# 실행은 fe_deapth에서 실행

# python3 T7_obb_pose.py 

# --------------------------------------------------------

# 폴더 구조

: << "END"

(rbcpc) recl3090@recl3090-Precision-3680:~/aavision/yolov11/f2_deapth$ ls
 d_test.py   T3_f3_yolo.py    'T3_pick point.py'   T7_obb_pose.py
 D_test.py   T3_mono_box.py    T5_3d_box.py        T8_g_datume_obb_pose.py
 T00.py      T3_multi_box.py   T6_obb.py           T9.py
(rbcpc) recl3090@recl3090-Precision-3680:~/aavision/yolov11/f2_deapth$ cd ..
(rbcpc) recl3090@recl3090-Precision-3680:~/aavision/yolov11$ ls
a_D435     proj                   ros_satting_txt  yolo11m.pt   yolov8n.pt
cam.py     realsense_depth_saves  runs             yolo11n.pt
f1_opencv  req.txt                test.py          yolo11s.pt
f2_deapth  robocup                use.txt          yolov11_env

END





