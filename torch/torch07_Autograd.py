# Autograd : 자동 미분 기능

# 경사하강법 : 비용 함수를 미분하여 이 함수의 기울기를 구하여 비용이 최소화 되는 방향으로 진행
 
import torch

w = torch.tensor(2.0, requires_grad = True)

y = w**2
z = 2*y + 5     # 2(w**2) + 5

z.backward()

print('수식을 w로 미분한 값 : {}'.format(w.grad))   # 수식을 w로 미분한 값 : 8.0
