"""
# 연속 분포 ( continuous distribution )
: 확률 밀도 함수 ( probability density function, pdf)로 표현한다.
   => 밀도 함수를 특정 구간에서 적분한 값으로 확률을 나타내는 함수 
"""

""" 균등 분포의 확률 밀도 함수"""
def uniform_pdf(x: float) -> flaot:
    return 1 if 0 <= x < 1 else 0


def uniform_cdf(x: float) -> float:
    """균등 분포를 따르는 확률변수의 값이 x보다 작거나 같은 확률을 변환"""
    if x < 0:   return 0    # 균등 분포의 확률은 절대로 0보다 작을 수 없다.
    elif x < 1: return x    # 예시: p(X <= 0.4) = 0.4
    else:       return 1    # 균등 분포의 확률은 항상 1보다 작다.


    
