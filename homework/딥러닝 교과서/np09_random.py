import numpy as np
'''
# np.random 모듈
: 난수 생성 가능
'''
# np.random()
from numpy.random import randint
import random

#1. np.random.rand() : 0 이상 1미만의 난수 생성
#                    : 넣은 정수의 횃수만큼 난수 생성
arr2 = np.random.rand(3)                              
print(arr2)                                          # [0.69152397 0.19592395 0.4505215 ]  

#2. np.random.randint(x, y, z) : x 이상 y 미만의 정수를 z 개 생성
#                              : z에 (2, 3)등의 인수를 넣을 수 있다. (2, 3)행렬 생성
arr1 = randint(0, 10, (5, 2))
print(arr1)                                          # [[8 6][6 5][1 8][8 3][6 5]] 
print(arr1.shape)                                    # (5, 2)

#3. np.random.normal(): 가우스 분포를 따르는 난수를 생성
arr3 = np.random.normal(5)
print(arr3)                                          # 4.570138857220005

