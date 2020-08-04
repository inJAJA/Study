import numpy as np


#1. 데이터
from tensorflow.keras.datasets import mnist
mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)                              # (60000, 28, 28)
print(x_test.shape)                               # (10000, )
print(y_train.shape)                              # (60000, )
print(y_test.shape)                               # (10000, )


# x_data전처리 : MinMaxScaler
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


# # y_data 전처리 : one_hot_encoding (다중 분류)
# from keras.utils.np_utils import to_categorical
# y_trian = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape)


# reshape : Dense형 모델 사용을 위한 '2차원'
x_train = x_train.reshape(60000, 28*28 ) 
x_test = x_test.reshape(10000, 28*28)
print(x_train.shape)                              # (60000, 784)
print(x_test.shape)                               # (10000, 784)



#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input

input_img = Input(shape=(784, ))
encoded = Dense(16, activation = 'relu')(input_img)     # 특성 추출 -> 잡음, 노이즈 제거 
                                                        # = 필요없는 값을 버린다. 0의 값(배경을 ) 없애준다.
                                                        # 중간 node 개수 정하는 법 : 샤머니즘?
                                                        #                           /너무 많이 주면 줄 필요가 없고, 너무 적게 주면 이미지가 흐려짐                                  
decoded = Dense(784, activation = 'sigmoid')(encoded)   # 전처리 해주었기 때문에 0 ~ 1사이의 값 가짐

autoencoder = Model(input_img, decoded)

autoencoder.summary()


#3. 훈련
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs = 50, batch_size = 256,                                  # 앞, 뒤가 똑같은
                            validation_split =0.2)

dencoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(dencoded_imgs[i].reshape(28,28)) # 비슷하게 나오는 이유는 배경(0)의 기여도가 낮기 때문이다.
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)   

plt.show()

