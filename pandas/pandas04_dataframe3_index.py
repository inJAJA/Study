"""DataFrame indexing"""
import pandas as pd
import numpy as np

data = {'names': ['A','A','A','B','B'],
        'year' : [2014, 2015, 2016, 2015, 2016],
        'points': [1.5, 1.7, 3.6, 2.4, 2.9]}

a = pd.DataFrame(data, columns = ['year','names','points','penalty'],
                       index = ['one','two','three','four','five'])

val = pd.Series([-1.2, -1.5, -1.7], index= ['two', 'four', 'five'])
a['debt'] = val

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



""" DataFrame에서 행을 선택하고 조작하기 """
# 0 ~ 2번째까지 가져온다.                  
print(a[0:3])
# Info   year names  points  penalty  debt
# Order                                   
# one    2014     A     1.5      0.1   NaN
# two    2015     A     1.7      0.2  -1.2
# thrid  2016     A     3.6      0.3   NaN


# 'two'라는 행부터 'four'라는 행까지 가져온다.
# 뒤에 써준 이름의 행을 빼지 않는다! -> 하지만 비추천
print(a['two':'four'])
# Info   year names  points  penalty  debt
# Order                                   
# two    2015     A     1.7      0.2  -1.2
# thrid  2016     A     3.6      0.3   NaN
# four   2015     B     2.4      0.4  -1.5


# 이 방법을 권장한다.
# .loc 또는 iloc함수 사용하는 방법 - 반환 형태는 Series
print(a.loc['two'])
# Info
# year       2015
# names         A
# points      1.7
# penalty     0.2
# debt       -1.2
# Name: two, dtype: object
# 'two'행의 값이 Series형태로 나온다. 

print(a.loc['two':'four'])
# Info   year names  points  penalty  debt
# Order                                   
# two    2015     A     1.7      0.2  -1.2
# thrid  2016     A     3.6      0.3   NaN
# four   2015     B     2.4      0.4  -1.5

print(a.loc['two':'four', 'points'])
# Order
# two      1.7
# thrid    3.6
# four     2.4
# Name: points, dtype: float64
# 'two'에서 'four'행까지의 'point'열을 불러온다.

print(a.loc[:,['year','names']])
# Info   year names
# Order            
# one    2014     A
# two    2015     A
# thrid  2016     A
# four   2015     B
# five   2016     B
# 전체행의 'year'과 'names'행을 가져 오겠다.

print(a.loc['three':'five','year':'penalty'])
# Info   year names  points  penalty
# Order                             
# three  2016     A     3.6      0.3
# four   2015     B     2.4      0.4
# five   2016     B     2.9      0.5
# 'three'에서 'five'까지의 행에서 
# 'year'에서 'penalty'까지의 값을 가져오겠다.

## [ : ,[ , ]] : list형태로 넣으면 각각의 값 나옴
## [ : ,  : ] : 구간으로 나옴


# 새로운 행 삽입하기
a.loc['six', : ] = [2013, 'jun', 4.0, 0.1, 2.1]
print(a)
# Info     year names  points  penalty  debt
# Order                                     
# one    2014.0     A     1.5      0.1   NaN
# two    2015.0     A     1.7      0.2  -1.2
# three  2016.0     A     3.6      0.3   NaN
# four   2015.0     B     2.4      0.4  -1.5
# five   2016.0     B     2.9      0.5  -1.7
# six    2013.0   jun     4.0      0.1   2.1


# .iloc사용
# : index 번호를 사용한다.
print(a.iloc[3])                # 3번째 행을 가져온다.
# Info
# year       2015
# names         B
# points      2.4
# penalty     NaN
# debt       -1.5
# Name: four, dtype: object


print(a.iloc[3:5, 0:2])        # 4 ~ 5행, 1 ~ 2열 반환
# Info     year names
# Order              
# four   2015.0     B
# five   2016.0     B

print(a.iloc[0:4, 2:4])        # 0 ~ 3행, 2 ~ 3 열 반환
# Info   points penalty  debt
# Order                      
# one       1.5     NaN   NaN
# two       1.7     NaN  -1.2
# three     3.6     NaN   NaN
# four      2.4     NaN  -1.5

# .iloc[a : b, c : d]   -> (a+1) ~  b행 까지
#                       ->  c ~ (d-1)열 까지


print(a.iloc[[0,1,3],[1,2]])
# Info  names  points          # 0, 1, 3행에서
# Order                        #    1, 2열 반환 
# one       A     1.5
# two       A     1.7
# four      B     2.4

"""
.loc  : 'key'값 사용
.iloc : 숫자(순서) 사용
"""