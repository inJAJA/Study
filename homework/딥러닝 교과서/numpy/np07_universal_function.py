'''
# 범용 함수 (universal function)
: ndarray 배열의 각 요소에 대한 연산결과 반환 함수
: 다차원 배열에도 사용 가능
'''
import numpy as np

a = np.array([2, 3, 5, 7, 9])
b = np.array([1, 2, 3, 4, 5])

# 1. 인수가 하나인 경우 
# np.abs()  : 절대값
a_abs = np.abs(a)
print(a_abs)                                           # [2 3 5 7 9]

# np.exp()  : 요소의 e(자연 로그의 밑)의 거듭제곱을 반환
a_exp = np.exp(a)
print(a_exp)                                           # [7.38905610e+00 2.00855369e+01 .... 8.10308393e+03]

# np.sqrt() : 제곱근 반환
a_sqrt = np.sqrt(a)
print(a_sqrt)                                          # [1.41421356 1.73205081 2.23606798 2.64575131 3.   ]


# 2. 인수가 두개인 경우
# np.add()  : 합 반환
a_add = np.add(a, b)                                   # [ 3  5  8 11 14]
print(a_add)

# np.substract() : 뺄셈
a_sub = np.subtract(a, b) 
print(a_sub)                                           # [1 1 2 3 4]

# np.maximum()   : 최댓값 반환
a_max = np.maximum(a, b)
print(a_max)                                           # [2 3 5 7 9]
