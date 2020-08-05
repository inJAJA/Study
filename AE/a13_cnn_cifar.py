# cifar10으로 autoencoder 구성할 것
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
import numpy as np

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'valid', 
                    input_shape= (32, 32, 3), activation = 'relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
    model.add(Conv2D(filters = hidden_layer_size, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
    model.add(Conv2DTranspose(filters = 156, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))  # Conv2D쓴만큼 Conv2DTranspose사용       
    model.add(Conv2DTranspose(filters = 300, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
    model.add(Conv2DTranspose(filters = 3, kernel_size = (3, 3), padding = 'valid', activation = 'sigmoid'))
 
    model.summary()

    return model

from tensorflow.keras.datasets import cifar10

train_set, test_set = cifar10.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

print(x_train.shape)
print(x_test.shape)

# x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 3))
# x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 3))

x_train = x_train/255.
x_test = x_test/255.


model = autoencoder(hidden_layer_size= 154)
                                                                                    
# model.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc'])                # loss: 8.7117e-05 - acc: 0.9133          
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])  # loss: 0.5470 - acc: 0.0121
                                                                                    

# model.fit(x_train_noised, x_train_noised , epochs = 10)     
model.fit(x_train, x_train, epochs = 10)


output = model.predict(x_test)

import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize = (20, 7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]], cmap = 'gray')
    if i ==0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]], cmap = 'gray')
    if i ==0:
        ax.set_ylabel('OUTPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()