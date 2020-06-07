import numpy as np
'''
# 전치 행렬 ( transposed matrix)
: 전치 = 행렬에서 행과 열을 바꾸는 것
'''
arr =  np.arange(10).reshape(2, 5)
print(arr.shape)                    # (2, 5)                         

# np.transpose()
arr1 = np.transpose(arr)
print(arr1.shape)                   # (5, 2)

# .T
arr2 = arr.T
print(arr2.shape)                   # (5, 2)

# 