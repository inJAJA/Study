"""
# 경사 하강법 ( gradient descent )
: 함수의 최댓값의 구하는 방법 중 하나는 임의의 시작점을 잡은 후,
 gradient를 계산하고, gradient의 방향(즉 함수의 출력값이 가장 많이 증가하는 방향)으로
 조금 이동한느 과정을 여러 번 반복하는 것이다.

:함수의 최솟값은 반대 방향으로 이동한다. 

# gradient (경사)
: 함수가 가장 빠르게 증가할 수 있는 방향을 나타낸다. 
"""
from scratch.linear_algebra import Vector, dot

def sum_of_squares(v: Vector) -> float:
    """v에 속해 있는 항목들의 제곱합을 계산한다."""
    return dot(v, v)