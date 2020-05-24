""" gradient구하기 """
from typing import Callable
def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h


""" 도함수( derivative) 구하기 """
# square 함수
def square(x: float) -> float:
    return x * x

def derivative(x: float) -> float:
    return 2 * x


xs =range(-10, 11)
actuals = [derivative(x) for x in xs]
estimates = [difference_quotient(square, x, h=0.001) for x in xs]


# 두 계산식의 결과값이 거의 비슷함을 보여 주기 위한 그래프
import matplotlib.pyplot as plt

plt.title("Actual Derivatives vs. Estimates")
plt.plot(xs, actuals, 'rx', label='Actual')               # 빨간색 x
plt.plot(xs, estimates, 'b+', label='Estimate')           # 파란색 +
plt.legend(loc =9)

plt.show()


"""편도 함수( partial derivative )"""
def partial_difference_quotient(f : Callable[[Vector], float],
                                v : Vector,
                                i: int,
                                h : float) -> float:
    """함수 f의 i번째 편도 함수가 v에서 가지는 값"""
    w = [v_j + (h if j == i else 0)            # h를 v의 i번쨰 변수에만 더해 주자
        for j, v_j in enumerate(v)]


"""일반적인 도함수과 같은 방법으로 gradient근사값 구하기"""
def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001):
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]
            