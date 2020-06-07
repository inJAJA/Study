import pandas as pd
'''열 추가'''
data = {'fruits':['apple','orange','banana','strawberry','kiwifruit'],
         'year':[2001, 2002, 2001, 2008, 2006],
         'time':[1, 4, 5, 6, 3]}          
df = pd.DataFrame(data)

# 열 추가
df['price'] = [150, 120, 100, 300, 150]
print(df)
#        fruits  year  time  price
# 0       apple  2001     1    150
# 1      orange  2002     4    120
# 2      banana  2001     5    100
# 3  strawberry  2008     6    300
# 4   kiwifruit  2006     3    150


# 문제
index = ['apple','orange','banana','strawberry','kiwifruit']
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
series1 = pd.Series(data1, index = index)
series2 = pd.Series(data1, index = index)
df = pd.DataFrame([series1, series2])

new_column = pd.Series([15, 7], index = [0, 1])
print(new_column)
# 0    15
# 1     7

# 열 추가
df['mango'] = new_column
print(df)
#    apple  orange  banana  strawberry  kiwifruit  mango
# 0     10       5       8          12          3     15
# 1     10       5       8          12          3      7