"""
# 신뢰구간 (confidence interval)
: 사건에 대한 분포를 모른다면 관축된 값에 대한 신뢰구간을 사용하여 가설 검증 가능
"""
from typing import Tuple

# 표준정규분포( standard normal distribution )
def normal_cdf(x: float, mu: float =0 , sigma: float =1) -> float:
    return (1 + math.erf((x-mu) / math.sqrt(2) / sigma)) /2

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

#--------------------------------------------------------

import math

# p의 정확한 값을 모른다면 추정값을 사용할 수 있다.
p_hat = 525 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)      
print(sigma)                                         # 0.0158

normal_two_sided_bounds(0.95, mu, sigma)             # [0.4940, 0.5560]


# 앞면이 540번 나왔다면
p_hat = 540 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)        # 0.0158

normal_two_sided_bounds(0.95, mu, sigma)             # [0.5091, 0.5709] 