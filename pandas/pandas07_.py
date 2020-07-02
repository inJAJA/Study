import pandas as pd
import numpy as np

# 열에 들어간 각 객체가 몇개씩 숫자 세기
a = pd.DataFrame({'A': [1, 2, 3, 4, 1, 3], 'B': ['q', 'w', 'e', 'r', 'w', 'r'], 'C':[51, 52, 53, 54, 51, 54]})
print(a)
#    A  B   C
# 0  1  q  51
# 1  2  w  52
# 2  3  e  53
# 3  4  r  54
# 4  1  w  51
# 5  3  r  54

a1 = a.A.value_counts(dropna = False)   # A에서
print(a1)
# 3    2
# 1    2
# 4    1
# 2    1
# Name: A, dtype: int64

a2 = a['A'].value_counts()              # AttributeError: 'Series' object has no attribute 'counts'
print(a2)
# 3    2
# 1    2
# 4    1
# 2    1
# Name: A, dtype: int64

print(pd.crosstab(a['A'], a['B']))      # pd.crosstab(index, columns) : 교차표 형성
# B  e  q  r  w
# A
# 1  0  1  0  1
# 2  0  0  0  1
# 3  1  0  1  0
# 4  0  0  1  0

b = a.drop('A', axis = 1)               # 한 열 버리기
print(b)
#    B   C
# 0  q  51
# 1  w  52
# 2  e  53
# 3  r  54
# 4  w  51
# 5  r  54