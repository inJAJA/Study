"""
# Data_Frame
 : 이전에는 DataFrame에 들어갈 data를 정의해주어야 하는데.
 : 이는 python의 dctionary 또는 numpy의 array로 정의할 수 있다.
"""
import pandas as pd
import numpy as np
data = {'name ':['a','b','c','d','e'],
        'year' : [2013, 2014, 2015, 2016, 2015],
        'points': [1.5, 1.7, 3.6, 2.4, 2.9]}

# DataFrame
# : 2차원
a = pd.DataFrame(data)
print(a)
#   name   year  points
# 0     a  2013     1.5
# 1     b  2014     1.7
# 2     c  2015     3.6
# 3     d  2016     2.4
# 4     e  2015     2.9
# 행과 열의 구조를 가진 데이터가 생긴다.


# 행 방향의 index                            # index : 행 
print(a.index)
# RangeIndex(start=0, stop=5, step=1)       # 5행을 가진다.


# 열 방향의 index                            # columns 열         
print(a.columns)                             
# Index(['name ', 'year', 'points'], dtype='object') 


# 값 얻기 
print(a.values)                             # data에 들어간 값 
# [['a' 2013 1.5]
# ['b' 2014 1.7]
# ['c' 2015 3.6]
# ['d' 2016 2.4]
# ['e' 2015 2.9]]


# 각 index에 대한 이름 설정하기
a.index.name = 'Num'                        # 행 이름
a.columns.name = 'Info'                     # 열 이름
print(a)
# Info name   year  points
# Num
# 0        a  2013     1.5
# 1        b  2014     1.7
# 2        c  2015     3.6
# 3        d  2016     2.4
# 4        e  2015     2.9


# DataFrame을 만들면서 columns와 index를 설정할 수 있다.
a2 = pd.DataFrame(data, columns=['year', 'name', 'points','penalty'],     
                   index = ['one', 'two','three','four','five'])
print(a2)
# DataFrame 을 정의 하면서, data로 들어가는 python dctionary와 columns의 순서가 달라도
# 알아서 맞춰서 정의 된다. (key를 보고서)
# 하지만 data에 포함되어 있지 않은 값은 NaN(Not a Number)으로 나타난다. (null과 같은 개념)
# 올바른 data처리를 위해 추가적으로 값을 넣어야 한다.
#        year name  points penalty
# one    2013  NaN     1.5     NaN
# two    2014  NaN     1.7     NaN
# three  2015  NaN     3.6     NaN
# four   2016  NaN     2.4     NaN
# five   2015  NaN     2.9     NaN


# describe() 
# : DataFrame의 계산 가능한 값들에 대한 다양한 계산 값을 보여준다.
print(a2.describe())
#               year    points
# count     5.000000  5.000000  :  데이터의 개수
# mean   2014.600000  2.420000  :  평균값
# std       1.140175  0.864292  :  표준편차 
# min    2013.000000  1.500000  :  최솟값
# 25%    2014.000000  1.700000  :  4분위수 25%
# 50%    2015.000000  2.400000  :         50% 
# 75%    2015.000000  2.900000  :         75%
# max    2016.000000  3.600000  :  최댓값



# DataFrame indexing
data = {'names': ['A','A','A','B','B'],
        'year' : [2014, 2015, 2016, 2015, 2016],
        'points': [1.5, 1.7, 3.6, 2.4, 2.9]}

a = pd.DataFrame(data, columns = ['year','names','points','penalty'],
                       index = ['one','two','three','four','five'])

print(a)
#        year names  points penalty
# one    2014     A     1.5     NaN
# two    2015     A     1.7     NaN
# thrid  2016     A     3.6     NaN
# four   2015     B     2.4     NaN
# five   2016     B     2.9     NaN


# dytpe
dtypes = a.dtypes
print(dtypes)
# year         int64
# names       object
# points     float64
# penalty     object


# astype
change_type = a.astype({'year':'float', 'penalty':'string'})
print(change_type.dtypes)
# year       float64
# names       object
# points     float64
# penalty     string
# dtype: object