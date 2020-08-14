# 이진 분류

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)                    # torch.Size([6, 2])
print(y_train.shape)                    # torch.Size([6, 1])

# W, b 정의
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


#optimizer
optimizer = optim.SGD([W, b], lr = 1)


# trian
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 정의
    y = x_train.matmul(W) + b

    # hypothesis = 1 / (1 + torch.exp(-(h)))  # sigmoid 적용
    hypothesis = torch.sigmoid(y)


    # loss 직접 계산
    '''
    losses = -(y_train * torch.log(hypothesis)                  # 모든 원소에 대한 오차
                + (1 - y_train) * torch.log(1 - hypothesis))
    print(losses)

    cost = losses.mean()
    print(cost)
    '''
    # binary_cross_entropy
    cost = F.binary_cross_entropy(hypothesis, y_train)


    # cost로 H(x)개선
    optimizer.zero_grad()                           # gradient 0으로 초기화
    cost.backward()                                 # gradient 계산
    optimizer.step()                                # W, b 업뎃

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))


hypothesis = torch.sigmoid(x_train.matmul(W) + b)   # 학습이 끝난 후의 W, b를 가지고 있음
print(hypothesis)                                   # tensor([[2.7648e-04],
                                                    #         [3.1608e-02],
                                                    #         [3.8977e-02],
                                                    #         [9.5622e-01],
                                                    #         [9.9823e-01],
                                                    #         [9.9969e-01]], grad_fn=<SigmoidBackward>)           

prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)                                   # tensor([[False],
                                                    #         [False],
                                                    #         [False],
                                                    #         [ True],
                                                    #         [ True],
                                                    #         [ True]])

print(W)                # tensor([[3.2530],
                        #         [1.5179]], requires_grad=True)
print(b)                # tensor([-14.4819], requires_grad=True)