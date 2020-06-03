import numpy as np
import pandas as pd

samsung = pd.read_csv('D:/Study/data/csv/samsung.csv', 
                     index_col = 0,                             # None
                     header = 0,                                # 0행을 header로 보겠다. / header = 1 : 0, 1이 header
                     sep = ',',
                     encoding = 'CP949')                        # 한글파일 처리 

hite = pd.read_csv('D:/Study/data/csv/hite.csv',
                   index_col = 0,                               # None
                   header = 0,
                   sep = ',',
                   encoding = 'CP949')

print(samsung)
print(hite.head())
print(samsung.shape)
print(hite.shape)
print(type(samsung))                                            # <class 'pandas.core.frame.DataFrame'>

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
samsung = samsung.dropna(axis =0)                                  # dropna = default ‘any’ : Nan이 들어간 모든 행을 삭제
# print(samsung)                                                   # axis = default 0 
print(samsung.shape)                                               # (509, 1)
hite = hite.fillna(method = 'bfill')                               # bfill = back fill  : 이후 행의 값으로 채워줌 ( 0행 <- 1행 )
hite = hite.dropna(axis = 0)                                       # ffill = front fill : 이전 행의 값으로 채워줌 ( 0행 -> 1행 )
# print(hite)


# Nan 제거 2
# hite = hite[0: 509]                                               # slicing으로 제거
# hite.iloc[0:, 1:5] = ['10','20','30','40']                        # [숫자]         : i = index, loc = location 
# hite.loc['2020-06-02', '고가':'거래량'] = ['10','20','30','40']      
print(hite)                                                         # ['index 이름'] : loc = location


# samsung과 hite의 정렬을 오름차순으로 변경
samsung = samsung.sort_values(['일자'], ascending = ['True'])
hite = hite.sort_values(['일자'], ascending = ['True'])

print(samsung)
print(hite)


# 콤마제거, 문자를 정수로 형변환
for i in range(len(samsung.index)):                                  # .index : 행   / .columns : 열
   samsung.iloc[i, 0] = int(samsung.iloc[i, 0].replace(',', ''))     # 모두 'str'형이 아니면 error뜬다  
   # samsung.iloc[i, 0] = samsung.iloc[i, 0].repalce(',','').astype(int)   
   # 기본 자료형(int, float, str 등)에 대해서는 stype을 적용할 수 없다.(dict, list, DataFrame에 대해서는 가능)
print(samsung)
print(type(samsung.iloc[0,0]))                                       # <class 'int'>

for i in range(len(hite.index)):                                     # '행' 반복
       for j in range(len(hite.iloc[i])):                            # '열' 반복
              hite.iloc[i, j] = int(hite.iloc[i, j].replace(',',''))

print(hite)
print(type(hite.iloc[1,1]))

print(samsung.shape)        # (509, 1)
print(hite.shape)           # (509, 5) 


# numpy로 변환 
samsung = samsung.values   
hite = hite.values

print(type(hite))           # <class 'numpy.ndarray'>

np.save('D:/Study/data/samsung.npy', arr = samsung )
np.save('D:/Study/data/hite.npy', arr = hite)

