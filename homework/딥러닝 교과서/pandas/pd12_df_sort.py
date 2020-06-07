import pandas as pd
import numpy as np
'''
# 정렬 
:오름차순으로 정렬
/ ascendong = False : 내림차순
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


# data 오름차순 정렬
df = df.sort_values(by = 'year', ascending= True)
print(df)
#        fruits  time  year
# 0       apple     1  2001
# 2      banana     5  2001
# 1      orange     4  2002
# 4   kiwifruit     3  2006
# 3  strawberry     6  2008

df = df.sort_values(by = ['time', 'year'], ascending= True)   # list 순서가 빠른 것 기준으로 우선정렬
print(df)
#        fruits  time  year
# 0       apple     1  2001
# 4   kiwifruit     3  2006
# 1      orange     4  2002
# 2      banana     5  2001
# 3  strawberry     6  2008


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


df = df.sort_values(by = columns)                            # list 순서가 빠른 것 기준으로 우선정렬
print(df)
#     apple  orange  banana  strawberry  kiwifruit
# 2       1       7      10           4         10
# 9       3       9       6           1          3
# 7       4       8       1           4          3
# 3       4       9       9           9          1
# 4       4       9      10           2          5
# 10      5       2       1           2          1
# 8       6       8       4           8          8
# 1       6       8       6           3         10
# 5       8       2       5           4          8
# 6      10       7       4           4          4