import numpy as np
import pandas as pd

datasets = pd.read_csv('./data/csv/iris.csv', # default  = None
                      index_col = None,      # None = index_column을 새로 생성, 기존 데이터에서 인덱스 column을 주지 않겠다.
                                             # 숫자 = 해당 열의 data사라지고 index_column생성됌
                      header = 0, sep=',')   # 0번째 행은 header로 보겠다.(데이터로 인식X) / None =  0행부터 data로 인식하겠다.    
#  header : defalut = 'infer'   # sep = seperate : 구분해주는 기준 / csv는 값이 ' , '로 나뉘어져 있다. 
#                     미루다     #                : default = ‘,’   
                                
#    index_column \  150    4  setosa  versicolor  virginica -> 0행 = header
#               0    5.1  3.5     1.4         0.2          0
#               1    4.9  3.0     1.4         0.2          0
#               2    4.7  3.2     1.3         0.2          0
#               3    4.6  3.1     1.5         0.2          0
#               4    5.0  3.6     1.4         0.2          0
#               ..   ...  ...     ...         ...        ...
#               145  6.7  3.0     5.2         2.3          2
#               146  6.3  2.5     5.0         1.9          2
#               147  6.5  3.0     5.2         2.0          2
#               148  6.2  3.4     5.4         2.3          2
#               149  5.9  3.0     5.1         1.8          2


#     index_column \   4  setosa  versicolor  virginica  -> 0행 = header
#               150
#               5.1  3.5     1.4         0.2          0
#               4.9  3.0     1.4         0.2          0
#               4.7  3.2     1.3         0.2          0
#               4.6  3.1     1.5         0.2          0
#               5.0  3.6     1.4         0.2          0
#               ..   ...     ...         ...        ...
#               6.7  3.0     5.2         2.3          2
#               6.3  2.5     5.0         1.9          2
#               6.5  3.0     5.2         2.0          2
#               6.2  3.4     5.4         2.3          2
#               5.9  3.0     5.1         1.8          2


print(datasets.shape) # index_col = None : (150, 5)   header = 0 
                      # index_col = 0    : (150, 4)   header = 0

                      # header = None    : (151, 5)   index_col = None
print(datasets)
'''
print(datasets.head())          # '위'에서 부터 5행만 보여준다.
print(datasets.tail())          # '아래'서 부터 5행만 보여준다.

print("=============")
'''
print(datasets.values)          # pandas를 numpy로 바꾼다.
'''
aaa = datasets.values
print(type(aaa))                # <class 'numpy.ndarray'>


# np로 저장
np.save('./data/iris_save.npy', arr = aaa)
'''