import numpy as np
'''
# 2차원 배열
: np.array([ ][ ])
'''
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# .shape : 차원의 요소수를 반환
print(arr.shape)                               # (2, 4)

# .reshape(a, b) : (a, b)의 모양의 행렬로 반환
arr1 = arr.reshape(4, 2)
print(arr1.shape)                               # (4, 2)


# index 참조
print(arr[1])                                   # [5 6 7 8]
print(arr[1, 2])                                # 7

# 슬라이스
print(arr[1, 1:])                               # [6 7 8]

# 문제
arr2 = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
print(arr2[0, 2])                               # 3
print(arr2[1:, :2])                             # [[4 5][7 8]]