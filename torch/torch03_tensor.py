import torch
import numpy as np

t = torch.tensor([[1, 2],[3, 4]])   # list 사용하여 tensor생성
print(t)                            # tensor([[1, 2], 
                                    #         [3, 4]])

# device
t = torch.tensor([[1, 2],[3, 4]], device='cuda:0')  # device : GPU에 tensor 생성
print(t)                                            # tensor([[1, 2],
                                                    #         [3, 4]], device='cuda:0')

# dtype
t = torch.tensor([[1, 2],[3, 4]], dtype=torch.float64)  # dtype : tensor의 데이터 형태 지정
print(t)                                                # tensor([[1., 2.],
                                                        #         [3., 4.]], dtype=torch.float64)

# .arange()
t = torch.arange(0, 10)             # arange를 이용한 1차원 tensor
print(t)                            # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# .to()
t = torch.zeros(3, 5).to('cuda:0')  # 모든 갑이 0인 3 x 5 tensor 작성 / to 메소드로 GPU에 전송
print(t)                            # tensor([[0., 0., 0., 0., 0.],
                                    #         [0., 0., 0., 0., 0.],
                                    #         [0., 0., 0., 0., 0.]], device='cuda:0')

# randn
t = torch.randn(3, 5)               # normal distribution으로 3 x 5 tensor 작성
print(t)                            # tensor([[-0.6414, -1.2244,  0.4100, -1.2775, -0.5613], 
                                    #         [-0.8621,  0.3671, -0.2415,  0.0263, -0.8288], 
                                    #         [-0.0247,  2.0675, -0.3721,  1.1948,  0.9889]])

# .size()
print(t.size())                     # tensor의 shape는 size 메서드로 확인
                                    # torch.Size([3, 5])
print(t.shape)                      # torch.Size([3, 5])

# .numpy()
t = torch.tensor([[1, 2],[3, 4]])   # numpy 사용하여 ndarray로 변환
x = t.numpy()
print(type(x))                      # <class 'numpy.ndarray'>

# .to('cpu').numpy()
t = torch.tensor([[1, 2],[3, 4]], device='cuda:0')  # GPU텐서 -> to메서드 -> CPU텐서 -> ndarray로 변환
x = t.to('cpu').numpy()
print(type(x))                                      # <class 'numpy.ndarray'>

# .linspace()
x = torch.linspace(0, 10, 5)        # 시작= 0, 끝= 10, step= 5
print(x)                            # tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])

y = torch.exp(x)
print(y)                            # tensor([1.0000e+00, 1.2182e+01, 1.4841e+02, 1.8080e+03, 2.2026e+04])

import matplotlib.pyplot as plt
plt.plot(x.numpy(), y.numpy())      # numpy와 호환되는 라이브도 사용 가능 
plt.show()                          # : torch -> numpy변환 가능하기 때문에