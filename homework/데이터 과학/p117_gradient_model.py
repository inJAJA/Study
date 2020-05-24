""" 경사 하강법으로 모델 학습 
: 경사 하강법으로 데이터에 적합한 모델의 파라미터를 구한다.
: 손실 함수(loss function)을 통해 모델이 얼마나 주어진 데이터에 적합한지 계산
"""
from scratch.linear_algebra import distance, add, scalar_multiply

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """v에서 step_size만큼 이동하기"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    retrun add(v, step)

#---------------------------------------------------------------
#     
# x는 -50 ~ 49 사이의 값이며, y는 항상 20 * x + 5
inputs = [(x, 20 * x + 5) for x in range(-50, 50)]


# 한 개의 데이터 포인터에서 오차의 gradient를 계산해 주는 함수
def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept           # 모델의 예측값
    error = (predicted - y)                     # 오차 = (예측값 - 실제값) 
    squared_error = error ** 2                  # 오차의 제곱을 최소화하자
    grad = [2 * error * x, 2 * error]           # gradient를 사용
    return grad


"""평균 제곱 오차( mean squared error )
 : 각 데이터 포인트에서 계산된 gradient의 평균

1. 임의의 theta로 시작
2. 모든 gradient의 평균을 계산
3. theta를 2번에서 계산된 값으로 변경
4. 반복
"""
from scratch.linear_algebra import vector_mean

# 임의의 경사와 절편으로 시작
import random
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

learning_rate = 0.001

for epoch in range(5000):
    # 모든 gradient의 평균을 계산
    grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
    # gradient만큼 이동
    theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1, "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"
