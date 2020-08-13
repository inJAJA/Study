import torch

model = nn.Linear(1, 1) # in = 1, out = 1

# 위의 모델을 class로 구성
class LinearRegressionModel(nn.Module):     # torch.nn.Module을 상속받는 파이썬 클래스
    def __init__(self):                     # 모델의 구조, 동적을 정의하는 생성자 정의
        super().__init__()                  # super() : nn.Module 클래스의 속성들을 가지고 초기화 됨
        self.linear = nn.Linear(1, 1)

    def forward(self, x):                   # 모델이 train data를 입력받아서 forward연산 진행
        return self.linear(x)               # model 객체를 데이터와 함께 호출하면 자동으로 실행

model = LinearRegressionModel()

#----------------------------------------------------------------------------------------------------

model = nn.Linear(3, 1)

class MultivariateLinearRegressinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegressinModel()
