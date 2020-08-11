import torch
'''
# 환경 설정
# conda create -n toto python==3.7 anaconda
1. python == 3.7.0
2. cuda == 10.0
3. cudnn == 10.0

# conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
=> pytorch == 1.6.0 (cuda 10.2 version)

# C++이 설치가 안되어 있어서 import시 errror 뜸
-> error 메세지에 뜬 주소에서 C++다운.설치 후 실행시 정상 작동 함
'''
print(torch.cuda.get_device_name(0))    # GeForce RTX 2080

print(torch.cuda.is_available())        # True : cuda 사용 여부

print(torch.__version__)                # 1.6.0

import torch.autograd   # 자동 미분을 위한 함수 포함
                        # enable_grad / no_grad: 자동 미분의 on, off를 제어
                        # Function : 자체 미분 가능 함수를 정의할 때 사용하는 기반 클래스

import torch.nn         # 신경망을 구축하기 위한 데이터 구조, 레이어 정의
                        # CNN, LSTM, activation, loss 등 정의

import torch.optim      # SGD 등의 파라미터 최적화 알고리즘 구현

import torch.utils.data # Gradient Descent 계열의 반복 연산을 할 때, 사용하는 미니 배치용 유틸리티 함수 포함

import torch.onnx       # ONNX(Oprn Neural Exchange) 포맷으로 모델 export 할 떄 사용
                        # ONNX는 서로 다른 딥러닝 프레임워크 간에 모델을 공유할 대 사용하는 새로운  포맷

                    