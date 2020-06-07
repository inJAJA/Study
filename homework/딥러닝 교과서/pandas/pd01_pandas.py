'''
# pandas 개요
: 데이터를 다루는 라이브러리
- Numpy : 데이터를 수학의 행렬로 처리할 수 있음 = 과학 계산에 특화
- Pandas : 일반적인 데이터베이스에서 이뤄지는 작업을 수행할 수 있다.
         : 수치뿐 아니라 이름과 주소 등 문자열 데이터도 쉽게 처리 가능

: pandas에는 Series와 DataFramse의 두 가지 데이터 구조가 존재
# Series
 : 1 차원 배열
 : Dataframe의 행 또는 열
 : 각 요소에 label이 붙어 있음 
# DataFrame
 : 2 차원 테이블 = 여러 Series를 묶은 것
 - 행 : 가로 방향의 데이터, label = index 
 - 열 : 세로 방향의 데이터, label = column
'''
import pandas as pd

# Series 
# : dictionary형을 전달하면 key에 의해 오름차순으로 정렬됨
fruits = {'orange': 2, 'banana': 3}
print(pd.Series(fruits))                                # orange    2
                                                        # banana    3
                                                        # dtype: int64]

# DataFrame
data = {'fruits':['apple','orange','banana','strawberry','kiwifruit'],
         'year':[2001, 2002, 2001, 2008, 2006],
         'time':[1, 4, 5, 6, 3]}          
df = pd.DataFrame(data)
print(df)                                               #        fruits  year  time
                                                        # 0       apple  2001     1
                                                        # 1      orange  2002     4
                                                        # 2      banana  2001     5
                                                        # 3  strawberry  2008     6
                                                        # 4   kiwifruit  2006     3

# Series용 label(index)
index = ['apple', 'orange', 'banana','strawberry','kiwifruit']

# Series data
data = [10, 5, 8, 12, 3]

# Series 작성
series = pd.Series(data, index = index)

# dictionary형 사용하여 dataframe작성
data = {'fruits':['apple','orange','banana','strawberry','kiwifruit'],
         'year':[2001, 2002, 2001, 2008, 2006],
         'time':[1, 4, 5, 6, 3]}    

# dataframe 작성
df = pd.DataFrame(data)

print('Series')
print(series)
print('\n')
print('DataFrame')
print(df)
# Series
# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwifruit      3
# dtype: int64


# DataFrame
#        fruits  year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3