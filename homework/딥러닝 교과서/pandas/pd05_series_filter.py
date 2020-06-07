import pandas as pd

'''
# 필터링
: Series형 데이터에서 조건과 일치하는 요소를 꺼내고 싶을때 사용
: bool형의 시퀀스를 지정해서 True인 것만 추출 가능
'''
index = ['apple', 'orange', 'banana','strawberry','kiwifruit']
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index = index)

# filter
conditions = [True, True, False, False, False]
print(series[conditions])
# apple     10
# orange     5
# dtype: int64

print(series[series >= 5])
# apple         10
# orange         5
# banana         8
# strawberry    12
# dtype: int64

series = series[series >= 5][series < 10]
print(series)
# orange    5
# banana    8
# dtype: int64