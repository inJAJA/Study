import numpy as np
'''
# 정렬
'''
arr = np.array([15, 30, 5])

# np.sort() : 정렬된 배열의 복사본 반환
print(np.sort(arr))                     # [ 5 15 30]

# .argsort() : 정렬된 배열의 인덱스(색인) 반환
print(arr.argsort())                    # [2 0 1]
print(np.argsort(arr))                  # [2 0 1]