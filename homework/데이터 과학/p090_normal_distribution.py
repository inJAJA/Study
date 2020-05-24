"""
# 정규 분포 ( normal distribution )
 : 종형 곡선 모양의 분포
 : 평균인 '뮤'와 표준편차 '시그마'의 두 파라미터 정의된다.
 : 평균은 종의 중심이 어디인지 나타냄, 표준편차는 종의 폭이 얼마나 넓은지 나타냄
"""
# 정규 분포의 밀도 함수
import math

SQRT_TWO_PI = math.sqrt(2 * math.pi)

def normal_pdf(x: float, mu: float =0 , sigma: float =1 ) -> float:
    return (math.exp(-(x-mu)** 2/ 2 / sigma ** 2  / (SQRT_TWO_PI * sigma)))


import matplotlib.pyplot as plt

xs = [x / 10.0 for x in range(-50, 50)]

plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-', label='mu=0,sigma=1')
plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '--', label='mu=0,sigma=2') 

plt.plot(xs,[normal_pdf(x, sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_pdf(x, mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend()
plt.title("Various Noraml pdfs")                  # 제목

plt.show()


""""
# 표준정규분포( standard normal distribution )
 : mu =0 , sigma =1인 정규분포를 의미
 
 : math.erf 
"""
def normal_cdf(x: float, mu: float =0 , sigma: float =1) -> float:
    return (1 + math.erf((x-mu) / math.sqrt(2) / sigma)) /2

xs = [x / 10.0 for x in range(-50, 50)]

plt.plot(xs,[normal_cdf(x, sigma=1)for x in xs], '-',label='mu=0,sigma=1')
plt.plot(xs,[normal_cdf(x, sigma=2)for x in xs], '--',label='mu=0,sigma=2')
plt.plot(xs,[normal_cdf(x, sigma=0.5)for x in xs], ':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x, mu=-1)for x in xs], '-.',label='mu=-1,sigma=1')

plt.legend(loc=4)                     # bottom right
plt.title("Various Nomarl cdfs")

plt.show()



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

    

