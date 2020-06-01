import numpy as np
import pandas as pd

datasets = pd.read_csv('./data/csv/iris.csv',
                      index_col = None,        # index_column은 비어있다.
                      header = 0, sep=',')     # 0번째 행은 hearder로 보겠다.(데이터로 인식X)       
                                # sep = seperate : 구분해주는 기준 / csv는 값이 ' , '로 나뉘어져 있다. 

print(datasets)

print(datasets.head())         # '위'에서 부터 5행만 보여준다.
print(datasets.tail())         # '아래'서 부터 5행만 보여준다.

print("=============")

print(datasets.values)         # pandas를 numpy로 바꾼다.

aaa = datasets.values
print(type(aaa))               # <class 'numpy.ndarray'>


# np로 저장
np.save('./data/iris_save.npy', arr = aaa)