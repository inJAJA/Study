# 행렬 연산을 고려

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# data 
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

print(x_train.shape)
print(y_train.shape)

# W, b 선언
W = torch.zeros((3, 1), requires_grad = True)   # (in, out)
b = torch.zeros(1, requires_grad = True)        # (out)

# optimizer
optimizer = optim.SGD([W, b], lr = 1e-5)


nb_epochs = 200

for epoch in range(nb_epochs + 1):

    # hypothesis
    h = x_train.matmul(W) + b

    # cost 계산
    cost = torch.mean((h - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} hypothesis: {}  Cost: {:.6f}'.format(
            epoch, nb_epochs, h.squeeze().detach(), cost.item()
        ))