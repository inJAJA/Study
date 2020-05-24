"""
# vector
 : vector끼리 더하거나 scalr와 곱해지면 새로운 벡터를 생성하는 개념적인 도수 
 : 여러 숫자의 list로 표현
"""
from typing import List

Vector = List[float]           # vector는 float객체를 갖고있는 list

height_weight_age = [70,       # 인치
                     170,      # 파운드
                     40]       # 나이

grades = [95,                  # 시험1 점수
          80,                  # 시험2 점수
          75,                  # 시험3 점수
          62]                  # 시험4 점수


def add(v: Vector, w: Vector) -> Vector:
    """각 성분끼리 더한다."""
    assert len(v) == len(w), "vectors must be same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


def subtract(v: Vector, w: Vector) -> Vector:
    """각 성분끼리 뺀다."""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]
    
assert subtract([5, 7, 9],[4, 5, 6]) == [1, 2, 3]   
    

def vector_sum(vectors: List[Vector]) -> Vector :
    """모든 벡터의 각 성분들끼리 더한다."""
    # vectors가 비어있는지 확인
    assert vectors, "no vectors provided"

    # 모든 벡터의 길이가 동일한지 확인
    num_elements = len(vectors[0])      # vectors[0] = vectors의 0번째 인자 
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # i번째 결과값은 모든 벡터의 i번째 성분을 더한 값
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

assert vector_sum([[1, 2],[3, 4],[5, 6],[7, 8]]) == [16, 20]


def scalar_multiply(c: float, v: Vector) -> Vector :
    """모든 성분을 c로 곱하기"""
    return [c * v_i for v_i in v]

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]


def vector_mean(vectors : List[Vector]) -> Vector:
    """각 성분별 평균은 계산"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([1, 2],[3, 4],[5, 6]) == [3 , 4]


"""백터의 내적( dot product )"""
def dot(v: Vector, w : Vector) -> float:
    """v_i * w_i + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3],[4, 5, 6]) == 32                # 1*4 + 2*5 + 3*6  


"""각 성분의 제곱의 합"""
def sum_of_squares(v: Vector) -> float:
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v , v)

assert sum_of_squares([1, 2, 3]) == 14               # 1*1 + 2*2 + 3*3


"""벡터의 크기 계산"""
import math

def magnitude(v: Vector) -> float :
    """벡터의 v의 크기를 반환"""
    return math.sqrt(sum_of_squares(v))              # math.sqrt() : 제곱근 게산

assert magnitude([3,4]) == 5


""" 두 벡터 간의 거리 계산 """
def squared_distance(v: Vector, w: Vector) -> float :
    """(v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    """벡터 v와 w간의 거리를 계산"""
    return math.sqrt(squared_distance(v, w))


# 한번에 정리한 식
def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))