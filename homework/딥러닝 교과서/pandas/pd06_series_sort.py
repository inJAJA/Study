import pandas as pd
'''
# 정렬
: 인수를 지정 안할 시 오름차순으로 정렬됌
/ ascending = False를 전달하면 내림차순 정렬
'''
index = ['apple', 'orange', 'banana','strawberry','kiwifruit']
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index = index)

# .sort_index() : index 정렬
items1 = series.sort_index()
print(items1)
# apple         10
# banana         8
# kiwifruit      3
# orange         5
# strawberry    12
# dtype: int64

# .sort_values() : data 정렬
items2 = series.sort_values()
print(items2)
# kiwifruit      3
# orange         5
# banana         8
# apple         10
# strawberry    12
# dtype: int64