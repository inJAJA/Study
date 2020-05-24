"""
# 중심 경향성( central tendency)
: 대부분 평균(average)를 사용
"""
from typing import List

num_friends = [100, 49, 41, 40, 25 ]              #... 등등 더 많은 데이터

def mean(xs: List[float]) -> float :
    return sum(xs) / len(xs)

print(mean(num_friends))                          # 51.0
 

"""
# 중앙값 (median)
 : 전체 데이터에서 가장 중앙에 있는 데이터 포인트
 : 데이터가 짝수일 경우 -  가장 중앙에 있는 두 데이터 포인트의 평균
"""
# 밑줄 표시로 시작하는 함수는 프라이빗 함수를 의미하며
# median 함수를 사용하는 사람이 직접 호출하는 것이 아닌
# median 함수만 호출하도록 생성되었다.

def _median_odd(xs: List[float]) -> float:
    """len(xs)가 홀수면 중앙값을 반환"""
    return sorted(xs)[len(xs)//2]

def _median_even(xs: List[float]) -> float:
    """len(xs)가 짝수면 두 중앙값의 평균을 반환"""
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2                     # length =4, hi_midpoint =2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2

def median(v: List[float]) -> float:
    """v의 중앙값을 계산"""
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

assert median([1, 10, 2, 9, 5]) == 5
assert median([1, 9, 2, 10])  == (2 + 9) / 2

print(median(num_friends))                         # 41


"""
# 분위( quantile )
: 중앙값을 포괄하는 개념
: 특정 백분위 보다 낮은 분위에 속하는 데이터를 의미한다.
"""
def quantile(xs : List[float], p: float) -> float:
    """x의 p분위에 속하는 값을 반환"""
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]

print(quantile(num_friends, 0.10))      # 25
print(quantile(num_friends, 0.25))      # 40
print(quantile(num_friends, 0.75))      # 49
print(quantile(num_friends, 0.90))      # 100


"""
# 최빈값( mode )
 : 데이터에서 가장 자주 나오는 값
"""
from collections import Counter

def mode(x: List[float]) -> List[float]:
    """최빈값이 하나보다 많을수도 있으니 결과를 리스트로 반환"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
                 if count == max_count]

x = [1, 1, 2, 3, 4, 5, 5, 5]
print(set(mode(x)))                    # {5}