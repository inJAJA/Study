import pandas as pd
import numpy as np
'''
# 데이터 참조
: 행과 열을 지정하여 참조 가능
- loc : '이름'으로 참조
- iloc : '번호'로 참조
'''
data = {'fruits':['apple','orange','banana','strawberry','kiwifruit'],
         'year':[2001, 2002, 2001, 2008, 2006],
         'time':[1, 4, 5, 6, 3]}          
df = pd.DataFrame(data)
print(df)
#        fruits  year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3


# .loc
df1 = df.loc[[1, 2], ['time', 'year']]
print(df)
#    time  year
# 1     4  2002
# 2     5  2001


# .iloc
df = df.iloc[[1, 3], [0, 2]]
print(df)
#        fruits  time
# 1      orange     4
# 3  strawberry     6


# 문제
np.random.seed(0)
columns = ['apple','orange','banana','strawberry','kiwifruit']

# dataframe생성 후 열 추가
df = pd.DataFrame()
for column in columns :
    df[column] = np.random.choice(range(1, 11), 10)

# range(시작행수, 종료행수 -1)
df.index = range(1, 11)
print(df)
#     apple  orange  banana  strawberry  kiwifruit
# 1       6       8       6           3         10
# 2       1       7      10           4         10
# 3       4       9       9           9          1
# 4       4       9      10           2          5
# 5       8       2       5           4          8
# 6      10       7       4           4          4
# 7       4       8       1           4          3
# 8       6       8       4           8          8
# 9       3       9       6           1          3
# 10      5       2       1           2          1


# .loc
df1 = df.loc[range(2, 6), ['banana','kiwifruit']]
print(df1)
#    banana  kiwifruit
# 2      10         10
# 3       9          1
# 4      10          5
# 5       5          8


# .iloc
df2 = df.iloc[range(1, 5), [2, 4]]
print(df2)
#    banana  kiwifruit
# 2      10         10
# 3       9          1
# 4      10          5
# 5       5          8