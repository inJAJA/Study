import pandas as pd
from pandas import DataFrame

dupli_data = DataFrame({'col1':[1, 1, 2, 3, 4, 4, 6, 6],
                        'col2': ['a','b','b','b','c','c','b','b']})

print(dupli_data)
#    col1 col2
# 0     1    a
# 1     1    b
# 2     2    b
# 3     3    b
# 4     4    c
# 5     4    c
# 6     6    b
# 7     6    b


''' .duplicated() '''
# 중복된 행을 True로 표시
print(dupli_data.duplicated())
# 0    False
# 1    False
# 2    False
# 3    False
# 4    False
# 5     True
# 6    False
# 7     True
# dtype: bool

''' .drop_duplicates() '''
# 중복 데이터가 삭제된 후의 데이터를 보여줌
print(dupli_data.drop_duplicates())
#    col1 col2
# 0     1    a
# 1     1    b
# 2     2    b
# 3     3    b
# 4     4    c
# 6     6    b