from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
import numpy as np

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'valid', 
                    input_shape= (28, 28, 1), activation = 'relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
    model.add(Conv2D(filters = hidden_layer_size, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
    model.add(Conv2DTranspose(filters = 156, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))  # Conv2D쓴만큼 Conv2DTranspose사용       
    model.add(Conv2DTranspose(filters = 300, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
    model.add(Conv2DTranspose(filters = 1, kernel_size = (3, 3), padding = 'valid', activation = 'sigmoid'))
 
    model.summary()

    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

x_train = x_train/255.
x_test = x_test/255.


# noise 추가
x_train_noised = x_train + np.random.normal(0, 0.1, size = x_train.shape) # mean(평균) = 0, std(표준편차) = 0.5
x_test_noised = x_test + np.random.normal(0, 0.1, size = x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min = 0, a_max = 1)  # np.clip() : min, max를 넘어가는 값을들 min, max값으로 바꿔줌
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max = 1)


model = autoencoder(hidden_layer_size= 154)
                                                                                    
# model.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc'])                
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])  
                                                                                    

# model.fit(x_train_noised, x_train_noised , epochs = 10)     
model.fit(x_train_noised, x_train, epochs = 10)


output = model.predict(x_test_noised)

import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize = (20, 7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('OUTPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()