import numpy as np
'''
# broadcast
: 크기가 다른 ndarray간의 연산에는 broadcast라는 처리가 자동으로 이루어 진다.
: 두 ndarray연산시 크기가 작은 배열의 행과 열을 자동으로 큰 쪽에 맞춰준다.
 - 행이 적은 쪽이 많은 쪽의 수에 맞추어 부족한 부분을 기존행에서 복사한다.
 ex )  [[0 1 2]   +  1        [[0 1 2]    + [[1 1 1]
        [3 4 5]]       =>      [3 4 5]]      [1 1 1]]
'''
x = np.arange(6).reshape(2, 3)
print(x + 1)                      # [[1 2 3]
                                  #  [4 5 6]]


x = np.arange(15).reshape(3, 5)
print(x)                          # [[ 0  1  2  3  4]
                                  #  [ 5  6  7  8  9]
                                  #  [10 11 12 13 14]]
print(x.shape)                    # (3, 5)

y = np.array([np.arange(5)])
print(y)                          # [[0 1 2 3 4]]
print(y.shape)                    # (1, 5)

z = x - y                         
print(z)                          # [[ 0  0  0  0  0]
                                  #  [ 5  5  5  5  5]
                                  #  [10 10 10 10 10]]


 

                                