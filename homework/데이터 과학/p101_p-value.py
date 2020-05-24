"""
# p-value
: 가설을 바라보는 관점
: 어떤 확률값을 기준으로 구간을 선택하는 대신에 
  H0가 참이라고 가정하고 실제로 관측된 값보다 더 극단적인 값이 나올 확률을 구하는 것 
"""
import math
from typing import Tuple

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

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)


#-----------------------------------------------------------------------


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


two_sided_p_value(529.5, mu_0, sigma_0)          # 0.062



# 시뮬레이션
import random

extreme_value_count = 0
for _ in range(1000):
    num_heads = sum(1 if random.random() < 0.5 else 0  # 앞면이 나온 경우를 세어 본다.
                    for _ in range(1000))              # 동전을 1000번 던져서
    if num_heads >= 530 or num_heads <= 470:
        extreme_value_count += 1                       # 몇 번 나오는지 세어 본다.


# p-value was 0.062 => ~ 63 evtreme values out of 1000
assert 59 < extreme_value_count < 65 , f"{extreme_value_count}" 
# 계산된 p-value가 5%가 크기 떄문에 귀무가설을 기각하지 않는다.


print(two_sided_p_value(531.5, mu_0, sigma_0))          # 0.0463


upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

# 동전 앞면이 525번 나왔다면 단측검정의 위한 p-value
print(upper_p_value(524.5, mu_0, sigma_0))              # 0.0606

# 동전 앞면이 527번 나왔다면 단측검정의 위한 p-value (귀무가설 기각)
print(upper_p_value(524.5, mu_0, sigma_0))              # 0.0047