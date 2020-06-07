import numpy as np
'''
# axis
: 좌표축
- axis = 0 : 열 마다 처리
- axis = 1 : 행 마다 처리
'''
arr = np.array([[1, 2, 3],[4, 5, 6]])
print(arr.sum())                       # 21       : scalar

# axis = 0
print(arr.sum(axis = 0))               # [5 7 9]  : 1d_arr

# axis = 1
print(arr.sum(axis = 1))               # [6 15]   : 1d_arr


arr1 = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
# axis =0
print(arr1.sum(axis = 0))              # [12 15 18]

# axis = 1
print(arr1.sum(axis = 1))              # [6 15 24]
