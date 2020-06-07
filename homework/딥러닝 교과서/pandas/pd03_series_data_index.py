import pandas as pd

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


''' data, index 추출 '''
# series.values
series_values = series.values
print(series_values)            # [10  5  8 12  3]

# series.index
series_index = series.index
print(series_index)             # Index(['apple', 'orange', 'banana', 'strawberry', 'kiwifruit'], dtype='object')