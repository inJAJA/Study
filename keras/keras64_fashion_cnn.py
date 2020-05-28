# 과제 2
# Sequential형으로 완성하시오.

# 하단에 주석으로 acc와 loss결과 명시하시오
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils.np_utils import to_categorical

#1. data
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)                                      # (60000, 28, 28)
print(x_test.shape)                                       # (10000, 28, 28)

# x : reshape
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print(x_train.shape)                                      # (60000, 28, 28, 1)

# y : one hot encoding
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
print(y_train.shape)                                      # (60000, 10)
print(y_test.shape)                                       # (10000, 10)
 


#2. model
model = Sequential()
model.add(Conv2D(100, (3, 3), input_shape = (28, 28, 1), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 3))
model.add(Dropout(0.2))
model.add(Conv2D(80, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 3))
model.add(Dropout(0.2))
model.add(Conv2D(60, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 3))
model.add(Dropout(0.2))
model.add(Conv2D(40, (3, 3), padding = 'same', activation = 'relu'))
model.add(Dropout(0.2))
model.add(Conv2D(20, (3, 3), padding = 'same', activation = 'relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))



#3. fit
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 50, batch_size = 64,
         validation_split = 0.2, shuffle = True, verbose =2 )


#4. evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size =64)
print('loss: ', loss)
print('acc: ', acc)

# acc:  0.9114999771118164
