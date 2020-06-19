import numpy as np

# ndarray = 배열을 고속으로 처리하는 class

# np.array( )
a = np.array([1, 2, 3])
print(a)                                              # [1 2 3]


""" Tensor """
# scalars : 1차원
a_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(a_1d.shape)                                    # (8,)

# matrix : 2차원
a_2d = np.array([[1, 2, 3, 4],[5, 6, 7, 8]])        
print(a_2d.shape)                                    # (2, 4)

# vector : 3차원
a_3d = np.array([[[1, 2], [3, 4]], [[5, 6],[7, 8]]])
print(a_3d.shape)                                    # (2, 2, 2)

# tensor : 4차원
a_4d = np.array([[[[1],[2]]],[[[3],[4]]]])
print(a_4d.shape)                                   # (2, 1, 2, 1)


''' numpy연산 '''
# 같은 위치에 있는 요소끼리 계산됌
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])

print(a + b)                                        # [2 4 6]
print(a - b)                                        # [0 0 0]
print(a * b)                                        # [1 4 9]
print(a / b)                                        # [1. 1. 1.]
print(a ** 3) # 3승                                 # [ 1  8 27]

# list
c = [1, 2, 3]
d = [1, 2, 3]

print(c + d)                                        # [1, 2, 3, 1, 2, 3]
# print(c - d)
# print(c * d)
# print(c % d)                                      # error


''' index 참조, slice '''
a = np.arange(10)
print(a)                                            # [0 1 2 3 4 5 6 7 8 9]

a1 = range(10)
print(a1)                                           # range(0, 10)

# slice
a[0 : 3] =1                                         # 0 ~2 까지
print(a)                                            # [1 1 1 3 4 5 6 7 8 9]


''' copy() '''
b = a.copy()                                        # a(ndarray)를 복사
print(b)                                            # [1 1 1 3 4 5 6 7 8 9]

b[0] = 3
print(b)                                            # [3 1 1 3 4 5 6 7 8 9]
c = b
print(c)                                            # [3 1 1 3 4 5 6 7 8 9]


''' bool index '''
