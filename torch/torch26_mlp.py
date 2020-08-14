import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


# data
X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)


# model
model = nn.Sequential(
    nn.Linear(2, 10, bias = True),
    nn.Sigmoid(),
    nn.Linear(10, 10, bias = True),
    nn.Sigmoid(),
    nn.Linear(10, 10, bias = True),
    nn.Sigmoid(),
    nn.Linear(10, 1, bias = True),
    nn.Sigmoid()
    ).to(device)                    # .to(device) : GPU사용


# loss function
criterion = torch.nn.BCELoss().to(device)   # nn.BCELoss() : 이진 분류에서 사용하는 cross entropy 함수

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 1)


# Train
for epoch in range(10001):
    optimizer.zero_grad()

    # forward 연사
    hypothesis = model(X)

    # 비용 함수
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()                # W, b 개선

    if epoch % 100 == 0:
        print(epoch, cost.item())


# prediction
with torch.no_grad():
    hypothesis = model(X)

    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    
    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    #     모델의 출력값(Hypothesis):  [[1.1174587e-04]
    #                                [9.9982870e-01]
    #                                [9.9984217e-01]
    #                                [1.8534536e-04]]
    
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    #     모델의 예측값(Predicted):  [[0.]
    #                                [1.]
    #                                [1.]
    #                                [0.]]
    
    print('실제값(Y): ', Y.cpu().numpy())
    #     실제값(Y):  [[0.]
    #                 [1.]
    #                 [1.]
    #                 [0.]]

    print('정확도(Accuracy): ', accuracy.item())    
    #      정확도(Accuracy):  1.0


