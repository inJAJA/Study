""" A와 B 중 어느 광고를 사용할지 선택해보자"""

# 표준정규분포( standard normal distribution )
def normal_cdf(x: float, mu: float =0 , sigma: float =1) -> float:
    return (1 + math.erf((x-mu) / math.sqrt(2) / sigma)) /2

# 만약 확률변수가 특정 값보다 작지 않다면, 특정 값보다 크다는 겂을 의미한다.
def normal_probability_above(lo: float,
                             mu : float = 0,
                             sigma: float = 1 ) -> float:
    """mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo보다 클 확률"""
    return 1 - normal_cdf(lo, mu, sigma)

def two_sided_p_value(x: float, mu: float =0 , sigma: float =1) -> float:
    """
    mu(평균값)와 sigma(표준편차)를 따르는 정규분포에서 x같이
    극단적인 값이 나올 확률은 얼마나 될까
    """
    if x >= mu:
        # 만약 x 가 평균보다 크다면 x 보다 큰 부분이 꼬리다.
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # 만약 x 가 평균보다 작다면 x 보다 작은 부분이 꼬리다.
        return 2 * normal_probability_above(x, mu, sigma)

#-------------------------------------------------------------------------

from typing import Tuple
import math

def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    p= n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma


# pA와 pB가 같다.(pA - pB =0)귀무가설 검정
def a_b_test_statistic(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)


z = a_b_test_statistic(1000, 200, 1000, 180)
print(z)                                         # -1.14

print(two_sided_p_value(z))                      # 1.745