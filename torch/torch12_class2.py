# 단순 선형 회귀 클래스로 구현하기

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# data
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)    # W, b 최적화

nb_epochs = 2000
for epoch in range(nb_epochs +1):

    prediction = model(x_train)

    cost = F.mse_loss(prediction, y_train)  # mse

    # cost로 H(x) 개선하는 부분
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0 :
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
