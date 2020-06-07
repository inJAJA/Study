import numpy as np
'''
# 통계 함수
: ndarray  배열 전체 또는 특정 축을 중심으로 수학적 처리는 수행하는 함수 or method
ex) .sum()
'''
arr = np.arange(15).reshape(3, 5)
print(arr)                         # [[ 0  1  2  3  4]
                                   # [ 5  6  7  8  9]
                                   # [10 11 12 13 14]]


# .sum() : 합게 반환
sum  = arr.sum(axis = 1)
print(sum)                         # [10 35 60]

# .mean() : 평균 반환
mean = arr.mean(axis  = 0)
print(mean)                        # [5. 6. 7. 8. 9.]

# np.average() : 평균 반환
average = np.average(arr)          # 7.0
print(average)

# np.max() : 최대값 반환
max = np.max(arr)
print(max)                         # 14

# np.min() : 최솟값 반환
min = np.min(arr)
print(min)                        # 0

# np.argmax() : 최댓값의 인덱스 번호 반환
argmax = np.argmax(arr, axis = 0)
print(argmax)

# np.argmin() : 최솟값의 인덱스 번호 반환

# np.std() : 표준편차 반환

# np.var() : 분산 반환
