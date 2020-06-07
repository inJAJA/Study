import pandas as pd
''' 행 추가 
# .append('Series형 데이터', ignore_index = True)
: df의 column과 df에 추가할 Series형 데이터의 index가 일치하지 않으면 
 df에 새로운 컬럼이 추가됌
 값이 존재하지 않는 요소는 NaN으로 채워줌
'''
data = {'fruits':['apple','orange','banana','strawberry','kiwifruit'],
         'time':[1, 4, 5, 6, 3]}          
df = pd.DataFrame(data)

# .append()
series = pd.Series(['mango', 2008, 70], index = ['fruits', 'year', 'time'])

df = df.append(series, ignore_index= True)
print(df)
#        fruits  time    year
# 0       apple     1     NaN
# 1      orange     4     NaN
# 2      banana     5     NaN
# 3  strawberry     6     NaN
# 4   kiwifruit     3     NaN
# 5       mango    70  2008.0


# 문제
index = ['apple','orange','banana','strawberry','kiwifruit']
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
data3 = [30, 12, 10, 8, 25, 3]
series1 = pd.Series(data1, index = index)
series2 = pd.Series(data2, index = index)
df = pd.DataFrame([series1, series2])
print(df)
#    apple  orange  banana  strawberry  kiwifruit
# 0     10       5       8          12          3
# 1     30      25      12          10          8

# series3 추가
index.append('pineapple')  
series3 = pd.Series(data3, index = index)
df = df.append(series3, ignore_index= True)
print(df)
#    apple  orange  banana  strawberry  kiwifruit  pineapple
# 0     10       5       8          12          3        NaN
# 1     30      25      12          10          8        NaN
# 2     30      12      10           8         25        3.0