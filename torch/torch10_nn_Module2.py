# 다중 선형 회귀

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)        # random seed 고정

# data
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]]) 


# model
model = nn.Linear(3, 1)     # in = 3 / out = 1

print(list(model.parameters()))

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)    # 학습 대상 : w, b

nb_epochs = 2000

for epoch in range(nb_epochs):

    # H(x) 개선
    prediction = model(x_train) # = model.forward(x_train)과 동일

    # loss
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()       # gradient를 0으로 초기화
    cost.backward()             # gradient 계산
    optimizer.step()            # W와 b를 업뎃

    # 100번마다 로그 출력
    if epoch % 100 == 0 :
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

# prediction
new_var =  torch.FloatTensor([[73, 80, 75]]) 

pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 
# 훈련 후 입력이 73, 80, 75일 때의 예측값 : tensor([[151.2305]], grad_fn=<AddmmBackward>)