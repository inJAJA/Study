import numpy as np
import random
# 연습 문제 
# 각 요소가 0~30인 정수 행렬(5 x 3)을 변수 arr에 대입하세요
arr = np.random.randint(0, 31, (5, 3))                      
print(arr)                                              # [[21 15 29]
                                                        #  [16 23 17]
                                                        #  [17 23  1]
                                                        #  [21 23 14]
                                                        #  [ 5  4  5]] 

# 변수 arr을 전치하세요
arr = arr.T
print(arr.shape)                                        # (3, 5)

# 변수 arr의 2, 3, 4열만 추출한 행렬 (3 x 3 )을 변수 arr1에 대입하세요
arr1 = arr[:, 2:].copy()
print(arr1)                                             # [[17 26 23]
                                                        #  [30  5 22]
                                                        #  [19  3 30]]

# 변수 arr1의 행을 정렬하세요
arr1 = np.sort(arr1)
print(arr1)                                             # [[17 23 26]
                                                        #  [ 5 22 30]
                                                        #  [ 3 19 30]]

# 각 열의 평균을 출력하세요
mean = arr1.mean(axis = 0)
print(mean)                                             # [ 8.33333333 21.33333333 28.66666667]


# 종합 문제
# 난수로 지정한 크기의 이미지를 생성하는 함수 make_image()를 완성하세요.
def make_image(n, m):
    image = np.random.randint(0, 6, (n, m))
    return(image)

image1 = make_image(3, 3)
print(image1)                                           # [[0 2 5] 
                                                        #  [3 5 5]
                                                        #  [1 5 1]]

# 전달된 행렬의 일부분을 난수로 변경하는 함수 change_matrix()를 완성하세요
def change_matrix(matrix):
    shape = matrix.shape
    for i in range(shape[0]):
        for j in range(shape[0]):
            if np.random.randint(0, 2) == 1:
                matrix[i][j] = np.random.randint(0, 6, 1)
    return matrix

image2 = change_matrix(image1.copy())
print(image2)                                           # [[0 2 4]
                                                        #  [3 5 2]
                                                        #  [5 4 1]]

# 생성된 image1과 image2의 각 요소의 하이의 절댓값을 계산하여  image3에 대입하세요
image3 = image1 - image2
image3 = np.abs(image3)
print(image3)                                           # [[0 0 1]
                                                        #  [0 0 3]
                                                        #  [4 1 0]]
