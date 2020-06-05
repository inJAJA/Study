import numpy as np
'''
# 인덱스 참조와 슬라이스는 리스트와 사용법 동일
# 1d_arr는  vector임으로 인덱스 참조한 곳은 scaler가 됌
'''
# slice
arr = np.arange(10)
print(arr)                                         # [0 1 2 3 4 5 6 7 8 9]

# index값 바꾸기
arr = np.arange(10)
arr[0:3] = 1
print(arr)                                         # [1 1 1 3 4 5 6 7 8 9]


arr = np.arange(10)
print(arr)                                         # [0 1 2 3 4 5 6 7 8 9]
 
print(arr[3:5])                                    # [3 4 5]

arr[2:5] = 24
print(arr)                                         # [ 0  1 24 24 24  5  6  7  8  9]