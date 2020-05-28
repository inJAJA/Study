import numpy as np

#1. 데이터
from keras.datasets import mnist
mnist.load_data()

(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# y : one hot encoding
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)

# x : minmax, reshape           # epoch = 5 / 40node레이어 없을 때 결과
# #1. (28*28, 1)
# x_train = x_train.reshape(60000, 28*28, 1).astype('float32') /255
# x_test = x_test.reshape(10000, 28*28, 1).astype('float32') /255

#2. (28, 28)
x_train = x_train.reshape(60000, 28, 28).astype('float32') /255
x_test = x_test.reshape(10000, 28, 28).astype('float32') /255
# acc:  0.641700029373169

# #3. (28*14, 2)
# x_train = x_train.reshape(60000, 28*14, 2).astype('float32') /255
# x_test = x_test.reshape(10000,28*14, 2).astype('float32') /255
# # acc:  0.11580000072717667

# #3. (28*7, 4)
# x_train = x_train.reshape(60000, 28*7, 4).astype('float32') /255
# x_test = x_test.reshape(10000,28*7, 4).astype('float32') /255
# # acc:  0.11349999904632568

# #4. (7*7, 4*4)
# x_train = x_train.reshape(60000, 7*7, 4*4).astype('float32') /255
# x_test = x_test.reshape(10000, 7*7, 4*4).astype('float32') /255
# # acc:  0.27140000462532043


#2. 모델
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (28, 28)))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(80, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax'))

model.summary()


#3. 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 10, batch_size =64,
          validation_split =0.2, shuffle = True, verbose = 2)

#4. 평가
loss, acc = model.evaluate(x_test, y_test, batch_size =64)
print('loss: ', loss)
print('acc: ', acc)

# acc:  0.9074000120162964