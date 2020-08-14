# class로 모델 구현

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)
print(y_train.shape)

'''
# model
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid
)
'''
# class로 model 구성 : 위와 동일
class BinaryClassifier(nn.Module):
    def __init__(self):                     # 구성
        super().__init__()                  # nn.Module 클래스의 속성 가져옴
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):                   # data받아서 forward연산
        return self.sigmoid(self.linear(x))

model = BinaryClassifier()


# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr = 1)


# train
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x)
    hypothesis = model(x_train)

    # cost
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x)개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 20 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])                     # 0.5 이상이면 True
        correct_prediction = prediction.float() == y_train                      # y_true와 일치하면 True
        accuracy = correct_prediction.sum().item() / len(correct_prediction)    # acc 계산

        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format( # 각 에포크마다 정확도를 출력
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))