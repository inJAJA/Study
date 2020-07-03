from scipy import special
import numpy as np

# Note
# numpy 배열 연산은 아주 빠르거나 아주 느릴 수 있는데, 이 연산을 빠르게 만드는 핵심은 바로 벡터화연산을 사용하는것
# 그것은 일반적으로 numpy의 Universal Function(ufunction)를 통해 구현된다.

# Note : Ufuncs에는 단일 입력값에 동작하는 단항 ufuncs,
#                    두 개의 입력값에 동작하는 이항 ufuncs가 있다. 

x = np.arange(4)

# Note : 배열 산술연산
print('x    = ', x)
print('x+5  = ', x + 5)
print('x-5  = ', x - 5)
print('x*2  = ', x * 2)
print('x/2  = ', x / 2)
print('x//2 = ', x // 2)  # 바닥 나눗셈(나머지는 버림)
print('-x   = ', -x)
print('x^2  = ', x**2)
print('x%2  = ', x % 2)

print('x    = ', x)
print('x+5  = ', np.add(x, 5))
print('x-5  = ', np.subtract(x, 5))
print('x*2  = ', np.multiply(x, 2))
print('x/2  = ', np.divide(x, 2))
print('x//2 = ', np.floor_divide(x, 2))
print('-x   = ', -x)
print('x^2  = ', np.power(x, 2))
print('x%2  = ', np.mod(x, 2))


# Note : 절댓값 함수
x = np.array([-2, -1, 0, 1, 2])
y = np.array([3-4j, 4-3j, 2+0j, 0+1j])

print(abs(x))
print(np.abs(x))
print(np.absolute(x))

print(abs(y))
print(np.abs(y))
print(np.absolute(y))


# Note : 삼각 함수
theta = np.linspace(0, np.pi, 3) # 0부터 pi가지 균등하게 3개의 원소로 구성된 배열을 만듦
print('theta       = ', theta)
print('sin(theta)  = ', np.sin(theta))
print('cos(theta)  = ', np.cos(theta))
print('tan(theta)  = ', np.tan(theta))
print('\n')

x = [-1, 0, 1]
print('x  = ', x)
print('arxsin(x)  = ', np.arcsin(x))
print('arccos(x)  = ', np.arccos(x))
print('arctan(x)  = ', np.arctan(x))


# Note : 지수와 로그함수
x = [1, 2, 3]
print('x    = ', x)
print('e^x  = ', np.exp(x))
print('2^x  = ', np.exp2(x))
print('3^x  = ', np.power(3, x))
print('\n')

x = [1, 2, 4, 10]
print('x      = ', x)
print('ln(x)    = ', np.log(x))
print('log2(x)  = ', np.log2(x))
print('log10(x) = ', np.log10(x))
print('\n') 


# Note : 매우 작은 입력값의 정확도 유지
x = [0, 0.001, 0.01, 0.1]
print('exp(x) - 1  = ', np.expm1(x))
print('log(x + 1)  = ', np.log1p(x))


# Note : 감마 함수(일반화된 계승)와 관련 함수
x = [1, 5, 10]
print('gamma(x)      = ', special.gamma(x))
print('ln|gamma(x)|  = ', special.gammaln(x))
print('beta(x)       = ', special.beta(x, 2))


# Note : 오차 함수(가우스 적분), 그 보수(complement)와 역수(inverse)
x = np.array([0, 0.3, 0.7, 1.0])
print('erf(x)    = ', special.erf(x))
print('erfc(x)   = ', special.erfc(x))
print('erfinv(x) = ', special.erfinv(x))

x = np.arange(5)
y = np.emtpy(5)
np.multiply(x, 10, out = y)
print(y)

y = np.zeros(10)
np.power(2, x, out = y[::2]) # ::n = n의 간격으로 
print(y)


# Note : 객체로부터 직접 연산하 수 있는 집계 함수
x = np.arange(1, 6)
print(np.add.reduce(x))          # 배열의 모든 요소의 합 반환
print(np.multiply.reduce(x))     # 배열의 모든 요소의 곱 반환

print(np.add.accumulate(x))      # 계산의 중간 결과를 모두 저장
print(np.multiply.accumulate(x))


# Note : 외적(outer products)
x = np.arange(1, 6)
np.multiply.outer(x, x)


# Note : 배열의 값의 합 구하기
L = np.random.random(100)
print(np.add.reduce(L))
print(np.sum(L))
sum(L)


# Note 
# 큰 배열에서의 sum메소드와 Numpy ufunction 배열 합 동작시간 비교
big_array = np.random.rand(1000000)
%timeit sum(big_array)
%timeit np.sum(big_array)


# Note : 최댓값과 최솟값
min(big_array), max(big_array)
np.min(big_array), np.max(big_array)


# Note
# 큰 배열에서의 min메소드와  Numpy ufunction 배열 최솟값 탐색 동작시간 비교
%timeit min(big_array)
%timeit np.min(big_array)

print(big_array.min(), big_array.max(), big_array.sum())


'''
# Numpy 집계 함수
np.sum, np.nansum(NaN안전모드) : 요소의 합 계산
np.prod, np.nanprod : 요소의 곱 계산
np.mean, np.nanmean : 요소의 평균계산
np.std, np.nanstd : 요소의 표준편차 계산
np.var, np.nanvar : 요소의 분산 계산
np.min, np.nanmin : 최솟값 계산
np.max, np.nanmax : 최댓값 계산
np.argmin, np.nanargmin : 최솟값의 인덱스 찾기
np.argmax, np.nanargmax : 최댓값의 인덱스 찾기
np.median, np.nanmedian : 요소의 중앙값 계산
np.percentile, np.nanpercentile : 요소의 순위 기반 백분위 수 계산
np.any : 요소 중 참이 있는지 검사
np.all : 모든 요소가 참인지 검사
'''


# Note : (열) axis = 0, (행) axis = 1
print(M.min(axis = 0))
print(M.min(axis = 1))

'''
# Note : 비교연산자와 대응 ufunc
== : np.equal
!= : np.not_equal
<  : np.less
<= : np.less_equal
>  : np.greater
>= : np.greater_equal
'''

rng = np.random.RandomState(0)
x = np.random.radint(10, size = (3, 4))

print(np.equal(x, 1))
print(np.not_equal(x, 1))
print(np.less(x, 5))
print(np.less_equal(x, 5))
print(np.greater(x, 5))
print(np.greater_equal(x, 5))

print(np.count_nonzero(np.less(x, 6)))
print(np.sum(np.less(x, 6)))
print(np.sum(np.less(x, 6), axis = 1))   # 각 행에 6보다 작은 값의 개수
print(np.sum(np.less(x, 6), axis = 0))   # 각 열에 6보다 작은 값의 갯수

print(np.any(np.greater(x, 8)))          # 8보다 큰 수가 있는가?
print(np.all(np.less(x, 10)))            # 모든 값이 10보다 작은가?
print(np.all(np.less(x, 8), axis = 1))   # 각 행의 모든 값이 8보다 작은가?

'''
# Note : bool 연산자와 대응 ufunc
& : np.bitwise_and
| : np.bitwise_or
^ : np.bitwise_xor
~ : np.bitwise_not
'''

# Note : 마스킹 연산
# 마스킹 연산 : 배열에서 조건에 맞는 값들을 선택

x[x < 5]
# output : array([4, 1, 2, 2, 1, 3])
# Note : 마스크 배열이 True인 위치에 있는 모든 값으로 채워짐