# nn.Module 사용

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# data
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)


# model
model = nn.Sequential(
    nn.Linear(2, 1),    # input_dim = 2, output_dim = 1
    nn.Sigmoid()        # sigmoid를 통과해서 나옴
)

model(x_train)


# optimizer 
optimizer = optim.SGD(model.parameters(), lr=1)


# train
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x)
    hypothesis = model(x_train)

    # cost 
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x)개선
    optimizer.zero_grad()   # gradient 0으로 초기화
    cost.backward()         # gradient 계산(역전파)
    optimizer.step()        # W, b 개선

    # 20번마다 로그 출력
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])                     # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train                      # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction)    # 정확도를 계산

        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(           # 각 에포크마다 정확도를 출력
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))

# predict
print(model(x_train))       # tensor([[2.7616e-04],
                            #         [3.1595e-02],
                            #         [3.8959e-02],
                            #         [9.5624e-01],
                            #         [9.9823e-01],
                            #         [9.9969e-01]], grad_fn=<SigmoidBackward>)

print(list(model.parameters()))
# [Parameter containing:
# tensor([[3.2534, 1.5181]], requires_grad=True), Parameter containing:     # W
# tensor([-14.4839], requires_grad=True)]                                   # b