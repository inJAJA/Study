import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)        # random seed 고정

# data
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

print(x_train)              
print(y_train)              

print(x_train.shape)        # torch.Size([3, 1])
print(y_train.shape)        # torch.Size([3, 1])


# 가중치 0으로 초기화
W = torch.zeros(1, requires_grad = True)    # requires_grad = True : 변수임을 명시
print(W)                                    # tensor([0.], requires_grad=True)

# 편향
b = torch.zeros(1, requires_grad = True)
print(b)                                    # tensor([0.], requires_grad=True)

# y = 0 * x + 0

'''
# hypothesis(가설) 세우기
h = x_train * W + b
print(h)


# loss
cost = torch.mean((h - y_train) ** 2)
print(cost)
'''

# Gradient Descent
optimizer = optim.SGD([W, b], lr = 0.01)

nb_epochs = 2000
for epoch in range(nb_epochs + 1 ):
    
    # H(x)
    h = x_train * W + b

    # cost
    cost = torch.mean((h - y_train) **2)

    # cost로 H(x) 개선
    optimizer.zero_grad()   # gradient을 0으로 초기화 : 새로운 가중치 편향에 대해서 새로운 기울기를 구할 수 있게 함
    cost.backward()         # 비용 함수를 미분하여 gradient(기울기) 계산
    optimizer.step()        # W, b를 없데이트 : 기울기에 lr을 곱하여 빼줌으로서 업데이트

    # 100번 마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
            ))

# Epoch 2000/2000 W: 1.997, b: 0.006 Cost: 0.000005