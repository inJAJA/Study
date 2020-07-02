"""DataFrame indexing"""
import pandas as pd
import numpy as np

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


"""DataFrame에서 열을 선택하고 조작하기"""
print(a['year'])
# one      2014
# two      2015
# thrid    2016
# four     2015
# five     2016
# Name: year, dtype: int64


# 동일한 의미를 갖는 다른 방법
print(a.year)
# one      2014
# two      2015
# thrid    2016
# four     2015
# five     2016
# Name: year, dtype: int64

print(a[['year', 'points']])
#        year  points
# one    2014     1.5
# two    2015     1.7
# thrid  2016     3.6
# four   2015     2.4
# five   2016     2.9


# 특정 열에 대해 위와 같이 선택하고, 우리가 원하는 값을 대입할 수 있다.
a['penalty'] = 0.5
print(a)
#        year names  points  penalty
# one    2014     A     1.5      0.5   penalty값이 0.5로 바뀜
# two    2015     A     1.7      0.5
# thrid  2016     A     3.6      0.5
# four   2015     B     2.4      0.5
# five   2016     B     2.9      0.5

a['penalty'] = [0.1, 0.2, 0.3, 0.4, 0.5] # python의 list나 numpy의 array
print(a)
#        year names  points  penalty
# one    2014     A     1.5      0.1
# two    2015     A     1.7      0.2
# thrid  2016     A     3.6      0.3
# four   2015     B     2.4      0.4
# five   2016     B     2.9      0.5


# 새로운 열 추가하기
a['zeros'] = np.arange(5)
print(a)
#        year names  points  penalty  zeros
# one    2014     A     1.5      0.1      0
# two    2015     A     1.7      0.2      1
# thrid  2016     A     3.6      0.3      2
# four   2015     B     2.4      0.4      3
# five   2016     B     2.9      0.5      4



# Series를 추가할 수도 있다.
val = pd.Series([-1.2, -1.5, -1.7], index= ['two', 'four', 'five'])
a['debt'] = val
print(a)
#        year names  points  penalty  zeros  debt  
# one    2014     A     1.5      0.1      0   NaN
# two    2015     A     1.7      0.2      1  -1.2
# thrid  2016     A     3.6      0.3      2   NaN
# four   2015     B     2.4      0.4      3  -1.5
# five   2016     B     2.9      0.5      4  -1.7
# index 'two','four','five'에 차례대로 값이 들어간다.
# 내가 지정한 index에 값을 넣을 수 있다는게 list나 array로 넣을 때와 큰 차이점


a['net_points'] = a['points'] - a['penalty']
a['high_points'] = a['net_points'] > 2.0
print(a)
#        year names  points  penalty  zeros  debt  net_points  high_points
# one    2014     A     1.5      0.1      0   NaN         1.4        False
# two    2015     A     1.7      0.2      1  -1.2         1.5        False
# thrid  2016     A     3.6      0.3      2   NaN         3.3         True
# four   2015     B     2.4      0.4      3  -1.5         2.0        False
# five   2016     B     2.9      0.5      4  -1.7         2.4         True
# 값이 알아서 계산되어 들어간다.


# 열 삭제하기
# del : delete
del a['high_points']
del a['net_points']
del a['zeros']
print(a)
#        year names  points  penalty  debt
# one    2014     A     1.5      0.1   NaN
# two    2015     A     1.7      0.2  -1.2
# thrid  2016     A     3.6      0.3   NaN
# four   2015     B     2.4      0.4  -1.5
# five   2016     B     2.9      0.5  -1.7
# 지정한 이름에 해당되는 열이 삭제 된다.


print(a.columns)
# Index(['year', 'names', 'points', 'penalty', 'debt'], dtype='object')


# 행, 열 이름 지정하기
a.index.name = 'Order'           # 행 이름
a.columns.name = 'Info'          # 열 이름
print(a)
# Info   year names  points  penalty  debt
# Order                                   
# one    2014     A     1.5      0.1   NaN
# two    2015     A     1.7      0.2  -1.2
# thrid  2016     A     3.6      0.3   NaN
# four   2015     B     2.4      0.4  -1.5
# five   2016     B     2.9      0.5  -1.7


print(a.shape)                   # (5, 5)
