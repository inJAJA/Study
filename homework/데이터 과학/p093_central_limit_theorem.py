import random
import math
import matplotlib.pyplot as plt

# 표준정규분포( standard normal distribution )
def normal_cdf(x: float, mu: float =0 , sigma: float =1) -> float:
    return (1 + math.erf((x-mu) / math.sqrt(2) / sigma)) /2


"""베르누이 확률변수(Bernoulli random variable)"""
def bernoulli_trial(p: float) -> int :
    """p의 확률로 1을, 1-p의 확률로 0을 반환"""
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    """n개 bernoulli(p)의 합을 반환"""
    return sum(bernoulli_trial(p) for _ in range(n))

# p = 베르누이의 확률변스의 평균 

from collections import Counter

def binomial_histogram(p: float, n: int, num_points: int) -> None:
    """binomial(n, p)의 결과값을 히스토그램으로 표현"""
    data = [binomial(n, p) for _ in range(num_points)]

    # 이항분포의 표본을 막대 그래프로 표현
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color = '0.75')

    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))

    # 근사된 정규분포를 라인 차트로 표현
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i -0.5, mu, sigma)
          for i in xs]
    
    plt.plot(xs, ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
    
    plt.show()


binomial_histogram(0.75, 100, 10000)