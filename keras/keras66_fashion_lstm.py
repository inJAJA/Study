# 과제 4
# Sequential형으로 완성하시오.

# 하단에 주석으로 acc와 loss결과 명시하시오
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout,LSTM
from keras.utils.np_utils import to_categorical

#1. data
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)
print(x_test.shape)


# y : one hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)
print(y_test.shape)

# x : reshape
x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test= x_test.reshape(x_test.shape[0], 28, 28)


#2. model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
model = Sequential()
model.add(LSTM(30, activation = 'relu', input_shape = (28, 28)))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(70, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax'))

model.summary()


#3. fit
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size =128,
          validation_split =0.2, shuffle = True, verbose = 2)

#4. evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size =128)
print('loss: ', loss)
print('acc: ', acc)

# acc:  0.47940000891685486
