""" 3차원 벡터의 최솟값 구하기"""
import random
from scratch.linear_algebra import distance, add, scalar_multiply

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """v에서 step_size만큼 이동하기"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    retrun add(v, step)


def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v ]


# 임의의 시작점을 선택
v = [random.uniform(-10, 10) for i in range(3)]

for epoch in range(1000):
    grad = sum_of_squares_gradient(v)               # v의 gradient 계산
    v = gradient_step(v, grad, -0.01)               # gradient의 음수만큼 이동
    print(epoch, v)

assert distance(v, [0, 0, 0]) < 0.001               # v는 0에 수렴해야 한다.