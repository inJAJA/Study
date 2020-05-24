""" 이항 분포는 정규 분포로 근사 할 수 있다."""
from typing import Tuple
import math

def normal_approximation_to_binomial(n : int, p: float) -> Tuple[float, float]:
    """Binomial(n, p)에 해당되는 mu(평균)와 sigma(표준편차) 계산"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

# 표준정규분포( standard normal distribution )
def normal_cdf(x: float, mu: float =0 , sigma: float =1) -> float:
    return (1 + math.erf((x-mu) / math.sqrt(2) / sigma)) /2


# 누적 분포 함수는 확률변수가 특정 값보다 작을 확률을 나타낸다.
normal_probability_below = normal_cdf


# 만약 확률변수가 특정 값보다 작지 않다면, 특정 값보다 크다는 겂을 의미한다.
def normal_probability_above(lo: float,
                             mu : float = 0,
                             sigma: float = 1 ) -> float:
    """mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo보다 클 확률"""
    return 1 - normal_cdf(lo, mu, sigma)


# 만약 확률변수가 hi보다 작고 lo보다 작지 않다면 확률변수는 hi와 lo 사이에 존재한다.
def normal_probability_between(lo: float,
                              hi: float,
                              mu: float = 0,
                              sigma: float =1) -> float:
    """mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo와 hi 사이에 있을 확률"""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)


# 만약 확률변수가 범위 밖에 존재한다면 범위 안에 존재하지 않다는 것을 의미한다.
def normal_probability_outside(lo: float,
                               hi: float,
                               mu: float =0,
                               sigma: float = 1) -> float:
    """mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo와 hi 사이에 없을 확률"""
    return 1 - normal_probability_between(lo, hi, mu, sigma)



"""분포의 60%를 차지하는 평균 중심의 구간을 구하기
: 양쪽 꼬이 부분이 각각 분포의 20%를 차치하는 지점을 구하면 된다."""

""" normal_cdf의 역함수 계산 """
def inverse_normal_cdf(p: float,
                       mu : float=0,
                       sigma: float=1,
                       tolerance: float = 0.00001) -> float:
    """이진 검색을 사용해서 역함수를 근사"""
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z = -10.0                     # normal_cdf(-10)은 0에 근접
    hi_z  =  10.0                     # normal_cdf( 10)은 1에 근접
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2    # 중간 값
        mid_p = normal_cdf(mid_z)     # 중간 값의 누적분포를 값을 계산
        if mid_p < p :       
            lw_z = mid_z              # 중간 값이 너무 작다면 더 큰 값들을 검색
        else:
            hi_z = mid_z              # 중간 값이 너무 크다면 더 작은 값들을 검색
    return mid_z

def normal_upper_bound(probability: float,
                       mu: float =0,
                       sigma: float =1 ) -> float:
    """P(Z <= z) = probability인 z 값을 반환"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float,
                       mu: float =0,
                       sigma: float =1 ) -> float:
    """P(Z >= z) = probability인 z 값을 반환"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float,
                            mu: float =0,
                            sigma: float =1) -> Tuple[float, float]:
    """
    입력한 probability값을 포함하고,
    평균을 중심으로 대칭적인 구간을 반환
    """
    tail_probability = (1 - probability) / 2

    # 구간의 상한은 tail_probability 값 이상의 확률 값을 갖고 있다.
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # 구간의 하한은 tail_probability 값 이하의 확률 값을 갖고 있다.
    lower_bound = normal_lower_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound


mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
print(mu_0, sigma_0)                  # 500.0, 15.811388300841896


"""
# 유의 수준(significance)
 : 제 1종 오류를 얼마나 허용해 줄것인가.
   - 제 1종 오류 : 비록 H0가 참이지만 H0를 기가하는 'flase positive(가양성)' 오류 
"""
lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)


"""
# 검정력(power)
: 제 2종 오류를 범하지 않을 확률을 구하면 검정력을 알 수 있다.
  - 제 2종 오류 : H0가 거짓이지만 H0를 기각하지 않는 오류를 의미
"""
# p가 0.5라고 가정할 때, 유의 수준이 5%인 구간
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)


# p = 0.55인 경우의 실제 평균과 표준편차
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)


# 제 2종 오류란 귀무가설(H0)을 기각하지 못한다는 의미
# 즉, X가 주어진 구간 안에 존재할 경우를 의미
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability                    # 0.887(?)


hi = normal_upper_bound(0.95, mu_0, sigma_0)
# 결과값은 526( < 531, 분포 상위 부분에 더 높은 확률을 주기 위해서)(?)


type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability                    #0.936(?)