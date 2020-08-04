# a06_ae 복붙


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units = 512, input_shape=(784, ), activation = 'relu'))
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dense(units = hidden_layer_size, activation = 'relu'))
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dense(units = 784, activation = 'sigmoid'))

    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

x_train = x_train/255.
x_test = x_test/255.

model = autoencoder(hidden_layer_size= 64)
                                                                                    # sigmoid를 썼기 때문에 'mse'를 사용해도 된다
# model.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc'])                # loss: 0.0102 -> 0.0041 
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])  # loss: 0.0935 -> 0.0744
                                                                                    # loss를 보고 결정해야함 / mse지표를 쓰면 acc의 값은 정확 X

model.fit(x_train, x_train , epochs = 10)

output = model.predict(x_test)

import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize = (20, 7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('INPUT', size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('OUTPUT', size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()