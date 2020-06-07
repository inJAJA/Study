import pandas as pd

fruits = {'banana': 3, 'prange': 2}
series = pd.Series(fruits)
print(series)
# banana    3
# prange    2
# dtype: int64

'''요소 추가'''
# Series에 요소를 추가하려면 해당 요소도 Series형이어야 함
series = series.append(pd.Series([3], index = ['grape']))
print(series)
# banana    3
# prange    2
# grape     3
# dtype: int64


# 문제
index = ['apple', 'orange', 'banana','strawberry','kiwifruit']
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index = index)
print(series)
# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwifruit      3
# dtype: int64

# 추가
pineapple = pd.Series([12], index = ['pineapple'])
print(pineapple)
# pineapple    12
# dtype: int64

series = series.append(pineapple)
print(series)
# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwifruit      3
# pineapple     12
# dtype: int64