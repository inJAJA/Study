import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist                          # keras에서 제공되는 예제 파일 

mnist.load_data()                                         # mnist파일 불러오기

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # mnist에서 이미 x_train, y_train으로 나눠져 있는 값 가져오기

print(x_train[0])                                         # 0 ~ 255까지의 숫자가 적혀짐 (그림)
print('y_train: ' , y_train[0])                           # 5

print(x_train.shape)                                      # (60000, 28, 28)
print(x_test.shape)                                       # (10000, 28, 28)
print(y_train.shape)                                      # (60000,)        : 10000개의 xcalar를 가진 vector(1차원)
print(y_test.shape)                                       # (10000,)


print(x_train[0].shape)                                   # (28, 28)
plt.imshow(x_train[59999], 'gray')                        # '2차원'을 집어넣어주면 수치화된 것을 이미지로 볼 수 있도록 해줌    
# plt.imshow(x_train[0])                                  # 색깔로 나옴
                                                          # plt.imshow를 두번 쓸 경우 맨 나중에 쓴 코드의 그림만 보여진다.

plt.show()                                                # 그림으로 보여주기

