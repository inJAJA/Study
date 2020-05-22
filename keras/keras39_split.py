import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1, 11))                     # a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]    (10, )
size = 65                                      #      0  1  2  3  4  5  6  7  8   9 

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):       # len = length  : 길이  i in range(6)  : [0, 1, 2, 3, 4, 5]
        subset = seq[i : (i + size)]           # i =0,  subset = a[ 0 : 5 ] = [ 1, 2, 3, 4, 5]
        aaa.append([item for item in subset])  # aaa = [[1, 2, 3, 4, 5]]
        #aaa.append([subset])
    print(type(aaa))
    return np.array(aaa)

# print(a)

# x = len(a)                                    # len(a) = 10
# print(x)

# aaa = []
# for i in range(len(a)- size +1):
#     print( 'i :', i)
#     subset = a[ i : (i+size)]
#     print(subset)
#     aaa.append([subset])
#     print(aaa)

dataset = split_x(a, size)
print("=================") 
print(dataset)   


