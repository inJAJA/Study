import pandas as pd
import numpy as np
'''
# 행 or 열 삭제
'''
data = {'fruits':['apple','orange','banana','strawberry','kiwifruit'],
          'time':[1, 4, 5, 6, 3],
          'year':[2001, 2002, 2001, 2008, 2006]}
df = pd.DataFrame(data)
print(df)
#        fruits  time  year
# 0       apple     1  2001
# 1      orange     4  2002
# 2      banana     5  2001
# 3  strawberry     6  2008
# 4   kiwifruit     3  2006

# .drop()
# 행 삭제
df_1 = df.drop(range(0, 2))
print(df_1)
#        fruits  time  year
# 2      banana     5  2001
# 3  strawberry     6  2008
# 4   kiwifruit     3  2006

# 열 삭제
df_2 = df.drop('year', axis = 1)
print(df_2)
#        fruits  time
# 0       apple     1
# 1      orange     4
# 2      banana     5
# 3  strawberry     6
# 4   kiwifruit     3


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


# 홀수 index만 삭제
df = df.drop(np.arange(2, 11, 2))     # 간격을 2로 생성
print(df)
#    apple  orange  banana  strawberry  kiwifruit
# 1      6       8       6           3         10
# 3      4       9       9           9          1
# 5      8       2       5           4          8
# 7      4       8       1           4          3
# 9      3       9       6           1          3

df = df.drop('strawberry', axis = 1)
print(df)
#    apple  orange  banana  kiwifruit
# 1      6       8       6         10
# 3      4       9       9          1
# 5      8       2       5          8
# 7      4       8       1          3
# 9      3       9       6          3

