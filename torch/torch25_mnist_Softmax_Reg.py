'''
## torchvision
: 유명한 데이터셋, 이미 구현되어 있는 유명한 모델들, 
  일반적인 이미지 전처리 도구들을 포함하고 있는 패키지
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.datasets as dsets
import torchvision.transforms as transform

import matplotlib.pyplot as plt
import random

# 사용 자원
USE_CUDA = torch.cuda.is_available()                      # GPU를 사용 가능하면 True, dkslaus False
device = torch.device('cuda' if USE_CUDA else 'cpu')      # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print('다음 기기로 학습합니다 :', device)                   # 다음 기기로 학습합니다 : cuda


# random seed 고정
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


# hyperparameters
training_epochs = 15
batch_size = 100


# MNIST dataset
mnist_train = dsets.MNIST(root='D:/data/MNIST_data/',           # Mnist 다운 받을 경로
                            train = True,                       # True = Train_data / False = Test_data    
                            transform=transform.ToTensor(),     # pytorch Tensor로 변환
                            download=True)                      # 해당 경로에 데이터가 없으면 다운
mnist_test = dsets.MNIST(root='D:/data/MNIST_data/',
                            train = False,
                            transform=transform.ToTensor(),
                            download=True)

# DataLoader
data_loader = DataLoader(dataset = mnist_train,
                            batch_size= batch_size,
                            shuffle = True,
                            drop_last = True)   # data가 batch_size로 딱 떨어지지 않았을 때 나머지 배치를 버림
                                                # 다른 미니 배치보다 개수가 적은 마지막 배치를 경사 하강법에 사용하여 
                                                # 마지막 배치가 상대적으로 과대 평가되는 현상을 막아줍니다.


# Model
linear = nn.Linear(784, 10, bias=True).to(device)   # GPU를 사용하기 위해서 .to(device)

# loss
criterion = nn.CrossEntropyLoss().to(device)    # softmax 포함

# optimizer
optimizer = torch.optim.SGD(linear.parameters(), lr = 0.1)


# Train
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:                # Mini-Batch 마다 훈련
        X = X.view(-1, 28 * 28).to(device)  # 배치 크기가 100이므로 아래의 연산에서 x는 (100, 784)의 Tensor가 된다.
        Y = Y.to(device)                    # One Hot Encoding 하지 않음

        
        # H(x)
        hypothesis = linear(X)

        # loss
        cost = criterion(hypothesis, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')