import numpy as np
'''
# 행렬 계산
'''
arr = np.arange(9).reshape(3, 3)

# np.dot(a ,b) : 두 행렬의 행렬 곱을 반환
dot = np.dot(arr, arr)
print(dot)                              # [[ 15  18  21][ 42  54  66][ 69  90 111]]
print(dot.shape)                        # (3, 3)

# np.linalg.norm(a) : norm을 반환
vector = arr.
norm = np.linalg.norm(arr)
print(norm)                             # 14.2828568570857


