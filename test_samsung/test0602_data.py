import numpy as np
import pandas as pd

samsung = pd.read_csv('D:/Study/data/csv/samsung.csv', 
                     index_col = 0,            # None
                     header = 0,               # 0행을 header로 보겠다. / header = 1 : 0, 1이 header
                     sep = ',',
                     encoding = 'CP949')       # 한글파일 처리 

hite = pd.read_csv('D:/Study/data/csv/hite.csv',
                   index_col = 0,              # None
                   header = 0,
                   sep = ',',
                   encoding = 'CP949')

print(samsung)
print(hite.head())
print(samsung.shape)
print(hite.shape)

'''
# 결측치를 채우는 법
1. 0으로 채운다
2. 평균값 (기간, 전체)
3. '이전' 값을 넣는다
4. '이후' 값을 넣는다
5. model을 만들어 predict값으로 넣어준다.(머신러닝 많이 씀)

=> 어떤 방법을 쓸 것인지는 감각으로 정한다.
   : 판단의 기준은 예측하여 나온 '정확도'를 보면된다.
'''

# Nan 제거 1
samsung = samsung.dropna(axis =0)              # dropna = default ‘any’ : Nan이 들어간 모든 행을 삭제
# print(samsung)                               # axis = default 0 
print(samsung.shape)                           # (509, 1)
hite = hite.fillna(method = 'bfill')           # bfill = back fill  : 이후 행의 값으로 채워줌 ( 0행 <- 1행 )
hite = hite.dropna(axis = 0)                   # ffill = front fill : 이전 행의 값으로 채워줌 ( 0행 -> 1행 )
# print(hite)


# Nan 제거 2
# hite = hite[0: 509]                          # slicing으로 제거
# hite.iloc[0:, 1:5] = ['10','20','30','40']        # [숫자]         : i = index, loc = location 
# hite.loc['2020-06-02', '고가':'거래량'] = ['10','20','30','40']      
print(hite)                                    # ['index 이름'] : loc = location

