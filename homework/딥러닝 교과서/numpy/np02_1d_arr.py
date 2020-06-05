# numpy 1D array
import numpy as np

# 1D array
list = np.array([1, 2, 3])
print(list)                                         # [1 2 3]

range = np.arange(4)
print(range)                                        # [0 1 2 3]


# 1D : vector
array_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# 2D : matrix
array_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# 3D : tensor
array_3d = np.array([[[1, 2, ], [3, 4], [5, 6]]])


storage = [24, 3, 4, 23, 10, 12]
print(storage)                                      # [24, 3, 4, 23, 10, 12]

np_storage = np.array(storage)
print(type(np_storage))                             # <class 'numpy.ndarray'>

'''
# 1d array 계산
# : 같은 위치에 있는 요소끼리 계산
'''
# list
storage = [1, 2, 3, 4]
new_storage =[]
for n in storage:
    n += n
    new_storage.append(n)
print(new_storage)                                  # [2, 4, 6, 8]

# numpy
storage = np.array([1, 2, 3, 4])
storage += storage
print(storage)                                      # [2 4 6 8]


arr = np.array([2, 5, 3, 4, 8])
print(arr + arr)                                    # [ 4 10  6  8 16]
print(arr - arr)                                    # [0 0 0 0 0]
print(arr**3)                                       # [  8 125  27  64 512]
print(1 /arr)                                       # [0.5        0.2        0.33333333 0.25       0.125  ]




