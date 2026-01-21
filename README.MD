🤖 RBCPC 가상환경 구축 및 실행 가이드
이 문서는 rbcpc 가상환경을 설정하고, YOLOv11 기반의 포즈 추정(Pose Estimation) 및 RealSense 카메라 연동 프로젝트를 실행하기 위한 매뉴얼입니다.

<video src="../thumbnail.mkv" width="100%" controls></video>

1. 가상환경 구축 (Conda)
먼저 기존 환경을 정리(선택 사항)하고, Python 3.9 기반의 새로운 환경을 생성합니다.

Bash

# (선택) 기존 환경 삭제
# conda env remove -n rbcpc

# 새 가상환경 생성
conda create -n rbcpc python=3.9 -y

# 가상환경 활성화
conda activate rbcpc
2. 필수 라이브러리 설치
프로젝트 실행에 필요한 기본 라이브러리를 설치합니다.

Bash

pip install pyrealsense2 numpy open3d opencv-python ultralytics
3. CUDA 및 PyTorch 설정
YOLOv11 및 GPU 가속을 사용하기 위해 시스템에 맞는 PyTorch를 설치해야 합니다.

3-1. CUDA 버전 확인
Bash

nvidia-smi
3-2. PyTorch 설치
본인의 그래픽 카드와 CUDA 버전에 맞는 PyTorch를 설치하세요.

참고: 설치 명령어는 PyTorch 공식 홈페이지에서 확인할 수 있습니다.

3-3. 설치 확인
설치가 완료되면 아래 명령어를 복사하여 GPU가 정상적으로 인식되는지 확인합니다.

Bash

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
✅ 정상 출력 예시:

PyTorch: 2.x.x+cu1xx CUDA Available: True Device Name: NVIDIA GeForce RTX 3090

4. 폴더 구조 (Directory Structure)
프로젝트는 아래와 같은 구조로 구성되어 있어야 합니다.

Plaintext

~/aavision/yolov11/
├── a_D435/
├── f2_deapth/              # (실행 작업 폴더)
│   ├── T3_pick point.py
│   ├── T5_3d_box.py
│   ├── T6_obb.py
│   ├── T7_obb_pose.py      # (메인 실행 파일)
│   └── ...
├── runs/
├── yolo11n.pt              # (모델 가중치 파일)
├── yolov8n.pt
└── ...
5. 프로젝트 실행
작업 디렉토리(f2_deapth)로 이동한 후 메인 스크립트를 실행합니다.

Bash

# 작업 폴더로 이동
cd ~/aavision/yolov11/f2_deapth

# OBB Pose Estimation 실행
python3 T7_obb_pose.py
📝 빠른 복사 (Quick Setup Snippet)
환경 설정부터 라이브러리 설치까지 한 번에 진행하려면 아래 블록을 사용하세요 (PyTorch 제외).

Bash

conda create -n rbcpc python=3.9 -y && conda activate rbcpc && pip install pyrealsense2 numpy open3d opencv-python ultralytics
