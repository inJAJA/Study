
# Dataset, DataLoader
# : 파이토치에서 데이터를 좀 더 쉽게 다룰 수 있는 도구
# : Mini Natch학습, 데이터 shuffle, 병렬 처리까지 간단히 수행 가능

# 사용 방법
# 1. Dataset을 정의
# 2. DataLoader에 전달하여 사용

import torch
import torch.nn as nn
import torch.nn.functional as F

# 임포트
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# data 
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])


# dataset으로 저장
dataset = TensorDataset(x_train, y_train)

# DataLoader
dataloader = DataLoader(dataset, batch_size= 2, shuffle = True)


# model
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)


# train
nb_epochs = 20
for epoch in range(nb_epochs +1):

    for batch_idx, samples in enumerate(dataloader):
        # print(batch_idx)
        # print(samples)
        # print('-----------------------')

        x_train, y_train = samples

        prediction = model(x_train)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))


# prediction
new_var =  torch.FloatTensor([[73, 80, 75]]) 

pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)
# 훈련 후 입력이 73, 80, 75일 때의 예측값 : tensor([[152.7419]], grad_fn=<AddmmBackward>)