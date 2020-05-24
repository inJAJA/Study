"""
# 산포도( dispersion )
: 데이터가 얼마나 펴져 있는지 나타냄
 - 0과 근접 : 데이터가 거의 펴져 있지 않다.
 - 큰 값    : 데이터가 매우 퍼져 있다.  
"""
num_friends = [100, 49, 41, 40, 25, 1, 4, 5, 10, 34]              

from typing import List

""" 범위 """
# 파이썬에서 "range"는 이미 다른것을 의미하기 떄문에 다른 이름을 사용하겠다.
def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)

print(data_range(num_friends))                     # 99


""" 분산(variance) """
def mean(xs: List[float]) -> float :
    return sum(xs) / len(xs)


from scratch.linear_algebra import sum_of_squares

def de_mean(xs: List[float]) -> List[float]:
    """x의 모든 데이터 포인트에서 평균을 뺌(평균을 0으로 만들기 위해)"""
    x_bar = mean(xs)
    return [ x- x_bar for x in xs]

def variance(xs: List[float]) -> float:
    """편차의 제곱근 (거의) 평균 """
    assert len(xs) >= 2, "variance requires at least two elements"

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)

print(variance(num_friends))


"""
# 표준 편차( standard_deviation )
 : 분산     = 기존 단위의 제곱
 : 표준편차 = 원래 단위와 같은 단위
"""
import math

def standard_deviation(xs: List[float]) -> float:
    """표준편차는 분산의 제곱근"""
    return math.sqrt(variance(xs))

print(standard_deviation(num_friends))



# 분위( quantile )
def quantile(xs : List[float], p: float) -> float:
    """x의 p분위에 속하는 값을 반환"""
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]


"""이상치가 주는 영향 제거"""
def interquartile_range(xs: List[float]) -> float:
    """상위 25%에 해당되는 값과 하위 25%에 해당되는 값의 차이를 반환"""
    return quantile(xs, 0.75) - quantile(xs, 0.25)

print(interquartile_range(num_friends))