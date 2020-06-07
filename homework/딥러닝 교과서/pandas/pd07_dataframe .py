import pandas as pd
'''
# DataFrame
: Series를 여러 개 묶은 것 같은 2차원 데이터 구조
: 리스트형의 길이가 동일해야 한다.
'''
# 생성 1
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

# 생성 2
index = ['apple','orange','banana','strawberry','kiwifruit']
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)

df2 = pd.DataFrame([series1, series2])
print(df2)
#    apple  orange  banana  strawberry  kiwifruit      
# 0     10       5       8          12          3      
# 1     30      25      12          10          8   