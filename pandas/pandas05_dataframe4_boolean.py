import pandas as pd
import numpy as np

data = {'year': [2014, 2015, 2016, 2015, 2016, 2013],
        'names': ['A','A','A','B','B','C'],
        'points': [1.5, 1.7, 3.6, 2.4, 2.9, 4.0],
        'penalty': [0.1, 0.2, 0.3, 0.4, 0.5, 0.1]}

a = pd.DataFrame(data, columns = ['year','names','points','penalty'],
                       index = ['one','two','three','four','five','six'])

debt = pd.Series([-1.2, -1.5, -1.7, 2.1], index = ['two','four','five','six'])
a['debt'] = debt

a.columns.name = 'Info'
a.index.name = 'Order'

print(a)
# Info   year names  points  penalty  debt
# Order
# one    2014     A     1.5      0.1   NaN
# two    2015     A     1.7      0.2  -1.2
# three  2016     A     3.6      0.3   NaN
# four   2015     B     2.4      0.4  -1.5
# five   2016     B     2.9      0.5  -1.7
# six    2013     C     4.0      0.1   2.1


"""4. DataFrame에서의 boolean Indexing"""
# year가 2014보다 큰 boolean data
print(a['year'] > 2014)
# Order
# one      False
# two       True
# three     True
# four      True
# five      True
# six      False
# Name: year, dtype: bool


# year가 2014보다 큰 모든 행의 값을 불러온다.
print(a.loc[a['year']>2014, :])
# Info   year names  points  penalty  debt
# Order
# two    2015     A     1.7      0.2  -1.2
# three  2016     A     3.6      0.3   NaN
# four   2015     B     2.4      0.4  -1.5
# five   2016     B     2.9      0.5  -1.7


# 'name'이 'A'인 값들의 'names'와 'points'값을 불러온다.
print(a.loc[a['names']=='A',['names','points']])
# Order
# one       A     1.5
# two       A     1.7
# three     A     3.6


# numpy에서와 같이 논리연산을 응용할 수 있다.

# 'point'가 2초과이고 'point'가 3미만인 행을 불러 오겠다.
print(a.loc[(a['points'] > 2) & (a['points'] < 3), : ])
print(a.loc[(a['points'] > 2) & (a['points'] < 3) ])
# Info   year names  points  penalty  debt
# Order
# four   2015     B     2.4      0.4  -1.5
# five   2016     B     2.9      0.5  -1.7


# 새로운 값을 대입할 수도 있다.
# 'point'가 3초과인 행의 'penalty'에 0을 집어 넣는다.
a.loc[a['points'] > 3, 'penalty'] = 0
print(a)
# Info   year names  points  penalty  debt
# Order
# one    2014     A     1.5      0.1   NaN
# two    2015     A     1.7      0.2  -1.2
# three  2016     A     3.6      0.0   NaN
# four   2015     B     2.4      0.4  -1.5
# five   2016     B     2.9      0.5  -1.7
# six    2013     C     4.0      0.0   2.1













