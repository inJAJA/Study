# 과제 3
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
print(x_train.shape)                   # (60000, 28, 28)
print(x_test.shape)                    # (10000, 28, 28)


# y : one hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)                   # (60000, 10)
print(y_test.shape)                    # (10000, 10)

# x : reshape
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test= x_test.reshape(x_test.shape[0], 28*28)


#2. model
model = Sequential()
model.add(Dense(200, activation = 'relu',  input_shape = (28*28, )))
model.add(Dropout(0.2))
model.add(Dense(150, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(80, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(70, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


# earlystopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 50, verbose = 1)

#3. fit
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 200, batch_size = 256, callbacks = [es],
          validation_split = 0.2, shuffle = True, verbose = 2)


#4. evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size = 256)
print('loss: ', loss)
print('acc: ', acc)

# acc:  0.8657000064849854