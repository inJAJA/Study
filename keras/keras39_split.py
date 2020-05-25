import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1, 11))                     # a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]    (10, )
                                               #      0  1  2  3  4  5  6  7  8   9 
size = 5                                       # 자르는 크기

# 함수를 사용하여 data_slpit
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):       # len = length  : 길이  i in range(6)  : [0, 1, 2, 3, 4, 5]
        subset = seq[i : (i + size)]           # i =0,  subset = a[ 0 : 5 ] = [ 1, 2, 3, 4, 5]
        aaa.append([item for item in subset])  # aaa = [[1, 2, 3, 4, 5]]
        #aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

# print(a)

# x = len(a)                                    # len(a) = a의 length
# print(x)                                      # 10           

# aaa = []                                      # 빈 list
# for i in range(len(a)- size +1):              # i in range( 10 - size + 1) = range(6) = [0, 1, 2, 3, 4, 5]
#     print( 'i :', i)                          # i = [0, 1, 2, 3, 4, 5]
#     subset = a[ i : (i+size)]                 # subset = a[0:5] / a[1:6] / a[2:7] / a[3:8] / a[4:9]
#     print(subset)
#     aaa.append([subset])
#     print(aaa)

dataset = split_x(a, size)                      # dataset.shape = ((len(a) - size +1) , size)
print("=================") 
print(dataset)   


