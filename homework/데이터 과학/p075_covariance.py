"""
# 공분산( covariance )
: 두 변수가 각각의 평균에서 얼마나 멀리 떨어져 있는지 살펴본다.
 (분산 
  : 하나의 변수가 평균에서 얼마나 떨어져 있는지 계산)
"""

num_friends = [100, 49, 41, 40, 25, 1, 5, 10, 34]  
daily_minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]


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


"""공분산 ( covariance )"""
from scratch.linear_algebra import dot

def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "xs and ys must have same number of elements"

    return dot(de_mean(xs), de_mean(ys)) / (len(xs)-1)

print(covariance(num_friends, daily_minutes))



# 표준 편차( standard_deviation )
import math

def standard_deviation(xs: List[float]) -> float:
    """표준편차는 분산의 제곱근"""
    return math.sqrt(variance(xs))



"""
# 상관관계( correlation )
 : 공분산에서 각각의 표준편차를 나눠 준 것
"""
def correlation(xs: List[float], ys: List[float]) -> float:
    """xs와 ys의 값이 각각의 평균에서 얼마나 멀리 떨어져 있는지 계산"""
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0                  # 편차가 존재하지 않는다면 상관관계 0

print(correlation(num_friends, daily_minutes))



"""이상치 ( outlier )"""
outlier = num_friends.index(10)   # 이상치의 인덱스

num_friends_good = [x
                    for i , x in enumerate(num_friends) 
                    if i != outlier]

daily_minutes_good = [x
                      for i, x in enumerate(daily_minutes)
                      if i != outlier]

daily_hours_good = [dm / 60 for dm in daily_minutes_good]

print(daily_minutes_good)
print(daily_hours_good)


"""
# Simpson's paradox
 : 혼재 변수(confounding variables)가 누락되어 상관관계가 잘못 계산되는 
 심슨의 역설
"""