"""
# 행렬( matrix ) 
: 2차원으로 구성된 숫자의 집합
: list의 list로 표현 가능
: list안의 list들은 행렬의 행( row )을 나타내며 모두 같은 길이를 가짐
"""
# 타입 명시를 위한 별칭
from typing import List

Vector = List[float]

Matrix = List[List[float]]

A = [[1, 2, 3],                   # A는 2개의 행과 3개의 열로 구성
     [4, 5, 6]]

B = [[1, 2],                      # A는 3개의 행과 2개의 열로 구성
     [3, 4],
     [5, 6]]


from typing import Tuple

def shape(A: Matrix) -> Tuple[int, int]:
    """(열의 개수, 행의 개수)를 반환"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0      # 첫 번째 행의 원소 개수
    return num_rows, num_cols

assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)      # 2 행, 3 열


def get_row(A: Matrix, i: int) -> Vector:
    """A의 i 번째 행을 반환"""
    return A[i]                           # A[i]는 i 번째 행을 나타낸다.

def get_column(A: Matrix, j: int) -> Vector:
    """A의 j 번째 열을 반환"""
    return [A_i[j]                        # A_i 행의 j 번째 원소
            for A_i in A]                 # 각 A_i 행에 대해


from typing import Callable

def make_matrix(num_rows : int,
                num_cols : int,
                entry_fn : Callable[[int, int], float]) -> Matrix:
    """
    (i, j)번쨰 원소가 entry_fn(i, j)인
    num_rows, num_cols 리스트를 반환
    """
    return [[entry_fn(i, j)               # i 가 주어졌을 떄, 리스드를 생성한다.
             for j in range(num_cols)]    # [entry_fn(i, 0), ...]
             for i in range(num_rows)]    # 각 i 에 대해 하나의 리스트를 생성


def identity_matrix(n: int) -> Matrix:
    """n x n 단위 행렬을 반환 """
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]] 

