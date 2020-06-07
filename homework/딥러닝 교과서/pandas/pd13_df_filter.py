import pandas as pd
import numpy as np
'''
# 필터링
: bool형의 시퀀스를 지정하여 True인 것만 추출하여 칠터링 수행
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

# filter
print(df.index % 2 == 0)          # bool형의 시퀀스 획득 
# [ True False  True False  True]

print(df[df.index % 2 == 0])      # 조건에 일치하는 요소를 포함하는 행을 가진 DataFrame생성
#       fruits  time  year
# 0      apple     1  2001
# 2     banana     5  2001
# 4  kiwifruit     3  2006


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


# filter 
df = df.loc[df['apple'] >= 5]
print(df)
#     apple  orange  banana  strawberry  kiwifruit
# 1       6       8       6           3         10
# 5       8       2       5           4          8
# 6      10       7       4           4          4
# 8       6       8       4           8          8
# 10      5       2       1           2          1

df = df.loc[df['kiwifruit'] >= 5]
print(df)
#    apple  orange  banana  strawberry  kiwifruit
# 1      6       8       6           3         10
# 5      8       2       5           4          8
# 8      6       8       4           8          8