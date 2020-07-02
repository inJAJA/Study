# pandas 불러오기
import numpy as np            # numpy도 함께 import
import pandas as pd

"""
# pandas 자료구조
: pandas에서 기본적으로 정의 되는 자료 구조인 Series와 Data frame을 사용합니다.
"""

# Series 정의
# : 1차원 배열의 형태
# : 한가지 기준에 의하여 데이터가 저장된다.
a = pd.Series([4, 7, -5, 3])
print(a) 
# 0   4
# 1   7
# 2  -5
# 3   3
# dtype: int64


# : Series의 값만 확인하기
print(a.values)
# [ 4  7 -5  3]


# : Series의 인덱스만 확인하기
print(a.index)
#RangeIndex(start=0, stop=4, step=1)


# Series의 자료형 확인하기
print(a.dtypes)
# int64


# 인덱스를 바꿀 수 있다.
a2 = pd.Series([4, 7, -5, 3], index = ['d', 'b', 'a', 'c'])
print(a2)
# d    4
# b    7
# a   -5
# c    3
# dtype: int64


# python의 dictionary 자료형을 Series data로 만들 수 있다.
# dictionary의 key가 Series의 index가 된다.
x = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
a3 = pd.Series(x)
print(a3)
# a    1
# b    2
# c    3
# d    4
# dtype: int64


# 이름 붙이기
a3.name = 'Salary'        
a3.index.name = "Names"     # index 이름 설정
print(a3)
# Names
# a    1
# b    2
# c    3
# d    4
# Name: Salary, dtype: int64


# index 변경
a3.index = ['q', 'w', 'e', 'r']
print(a3)
# q    1
# w    2
# e    3
# r    4
# Name: Salary, dtype: int64