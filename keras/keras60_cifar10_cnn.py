# cifar10 색상이 들어가 있다.
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np

#1. data
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

print(x_train[0])
print('y_train[0] :',y_train[0])

print(x_train.shape)                # (50000, 32, 32, 3)
print(x_test.shape)                 # (10000, 32, 32, 3)
print(y_train.shape)                # (50000, 1)
print(y_test.shape)                 # (10000, 1)


# x : reshape, minmax 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

x = np.vstack([x_train, x_test])

print(x.shape)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0])

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
print(x_train[0,0,0,:])

"""
이렇게 minmax를 할 경우에 유실되는 데이터가 있을 수 있으니 샘플을 뽑아서 확인해보는게 좋다.
data가 좋지 않아 이상치 or 결실치가 많을 경우에는 효과적이지만
이상적인 data에서는 별로 영향이 없다.
"""


# y : one hot categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)                 # (50000, 10)


#2. model
input1 = Input(shape = (32, 32, 3))

dense1 = Conv2D(200, (3, 3), activation = 'relu',padding = 'same')(input1)
max1 = MaxPooling2D(pool_size = 2)(dense1)
drop1 = Dropout(0.3)(max1)

dense2 = Conv2D(100, (3, 3), activation = 'relu',padding = 'same')(drop1)
max2= MaxPooling2D(pool_size = 2)(dense2)
drop2 = Dropout(0.2)(max2)


dense3 = Conv2D(80, (3, 3), activation = 'relu',padding = 'same')(drop2)
max3 =  MaxPooling2D(pool_size = 2)(dense3)
drop3 = Dropout(0.2)(max3)

dense4 = Conv2D(40, (3, 3), activation = 'relu', padding = 'same')(drop3)
drop3 = Dropout(0.2)(dense4)

dense5 = Conv2D(20, (3, 3), activation = 'relu')(drop3)

flat = Flatten()(dense5)
output1 = Dense(10, activation = 'softmax')(flat)

model = Model(inputs = input1, outputs = output1)

model.summary()


#3. fit
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs= 140, batch_size = 256,
              validation_split =0.2, shuffle = True, verbose =2)


#4. evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size = 256)
print('loss: ', loss)
print('acc: ', acc)

# acc:  0.8070999979972839