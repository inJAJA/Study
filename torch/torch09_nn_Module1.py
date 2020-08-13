# nn.Module을 이용하여 모델 만들기
'''
import torch.nn as nn
model = nn.Linear(input_dim, output_dim)    # 선형 회귀 모델

import torch.nn.functional as F
cost = F.mse_loss(prediction, y_train)      # mse
'''
#-----------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)            # random seed 고정

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

# model
model = nn.Linear(1, 1)         # input_dim = 1 / output_dim = 1

print(list(model.parameters())) # w, b값 출력

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

# 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)  # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    optimizer.zero_grad()                   # gradient를 0으로 초기화
    cost.backward()                         # backward 연산, 비용 함수를 미분하여 gradient 계산 
    optimizer.step()                        # W와 b를 업데이트

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

# prediction
new_var = torch.FloatTensor([[4.0]])

pred_y = model(new_var) # forward 연산 : 예측값 리턴
print('훈련 후 입력이 4일 때의 예측값 :', pred_y)         # 훈련 후 입력이 4일 때의 예측값 
                                                        # : tensor([[7.9989]], grad_fn=<AddmmBackward>)

print(list(model.parameters()))
# [Parameter containing:
# tensor([[1.9994]], requires_grad=True), Parameter containing:
# tensor([0.0014], requires_grad=True)]
