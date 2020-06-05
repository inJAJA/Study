# numpy 
# : 파이썬으로 벡터나 행렬 계산을 빠르게 하도록 특화된 기본 라이브러리

# 라이브러리 : 외부에서 읽어 들이는 파이썬 코드 묶음

# 힐요한 라이브러리 import
import numpy as np
import time
from numpy.random import rand                             

# 행, 열 크기
N =150

# 행렬을 초기화합니다
matA = np.array(rand(N, N))
matB = np.array(rand(N, N))
matC = np.array([[0]*N for _ in range(N)])

# 파이썬의 list를 사용하여 계산
# 시작 시간을 저장
start = time.time()

# for문을 사용하여 행렬 곱셈을 실행
for i in range(N):
    for j in range(N):
        for k in range(N):
            matC[i][j] = matA[i][k] * matB[k][j]

print("파이썬 기능만으로 계산한 결과 : %.2f[sec]" % float(time.time() - start))

# numpy를 사용하여 계산합니다.
# 시작 시간을 저장합니다.
start = time.time()

# numpy를 사용하여 행렬 곱셈을 실행합니다.
matC = np.dot(matA, matB)

# 소수점 아래 두 자리까지 표시됨
# numpy는 0.00[sec]로 표시됌
print("Numpy를 사용하여 계산한 결과 : %.2f[sec]"% float(time.time() - start))
# 파이썬 기능만으로 계산한 결과 : 3.45[sec]
# Numpy를 사용하여 계산한 결과 : 0.03[sec]

