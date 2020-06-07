import pandas as pd
'''
# Series 
: 1 차원 배열처럼 다룰 수 있다.
: index를 지정하지 않으면 0부터 순서대로 정수 index가 붙음
'''
fruits = {'banana': 3, 'orange': 2}
print(pd.Series(fruits))
# banana    3
# orange    2
# dtype: int64


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


''' 참조 '''
fruits = {'banana':3, 'orange': 4, 'grape': 1, 'peach': 5}
series2 = pd.Series(fruits)

print(series2[0:2])
# banana    3
# orange    4
# dtype: int64

print(series2[['orange', 'peach']])
# orange    4
# peach     5
# dtype: int64

items1 = series[2:]
print(items1)
# banana         8
# strawberry    12
# kiwifruit      3
# dtype: int64

items2 = series[['apple','banana','kiwifruit']]
print(items2)
# apple        10
# banana        8
# kiwifruit     3
# dtype: int64





