"""
# 미니배치 경사 하강법( minibatch ) 
  - 경사 하강법의 단점 
  : 데이터셋 전체의 gradient를 모두 구해야 이동거리만큼 파라미터를 업데이트 할수 있음
: 전체 데이터셋의 sample인 minibatch에서 gradient계산
  ( 큰 데이터샛 모델을 하긋ㅂ하는 경우에 유리 )
"""
inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

from scratch.linear_algebra import distance, add, scalar_multiply

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """v에서 step_size만큼 이동하기"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    retrun add(v, step)

# 한 개의 데이터 포인터에서 오차의 gradient를 계산해 주는 함수
def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept           # 모델의 예측값
    error = (predicted - y)                     # 오차 = (예측값 - 실제값) 
    squared_error = error ** 2                  # 오차의 제곱을 최소화하자
    grad = [2 * error * x, 2 * error]           # gradient를 사용
    return grad

learning_rate = 0.001

#-----------------------------------------------------------------------------

from typing import TypeVar, List, Iterator
import random

T = TypeVar('T')                          # 변수의 타입과 무관한 함수를 생성

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = Ture) -> Iterator:
    """dataset에서 batch_size만큼 데이터 포인터를 샘플링해서 minibatch를 생성"""
    # 각 minibatch의 시작점인 0, batch_size, 2 * batch_size, ...을 나열
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts)     # minibatch의 순서를 섞는다. 

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start : end]

#-------------------------------------------------------------------
""" minibatch로 다시 풀어보자 """

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(1000):
    for batch in minibatches(inputs, batch_size=20):
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1, "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"

#--------------------------------------------------------------------
""" SGD ( stochastic gradient descent )
 : 각 epoch 마다 단 하나의 데이터 포인트에서 gradient를 계산한다.
"""

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(100):
    for x, y in inputs:
        grad = linear_gradient(x, y, theta)
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1, "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"
# SGD가 훨씬 적은 epoch안에서 최적의 파라미터를 찾아낸다datetime A combination of a date and a time. Attributes: ()

"""
# 단점

minibatch : 더 오래 걸림
SGD       : 특정 데이터 포인트의 gradient와 
            데이터셋 전체의 gradient의 방향이 서로 상반될 수도 있다.
"""   