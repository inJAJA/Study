# cifar10 색상이 들어가 있다.
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

#1. data
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

print(x_train[0])
print('y_train[0] :',y_train[0])

print(x_train.shape)                # (50000, 32, 32, 3)
print(x_test.shape)                 # (10000, 32, 32, 3)
print(y_train.shape)                # (50000, 1)
print(y_test.shape)                 # (10000, 1)

# x : reshape, minmax 
x_train = x_train.reshape(x_train.shape[0], 32*32*3)
x_test = x_test.reshape(x_test.shape[0], 32*32*3)
print(x_train.shape)


# y : one hot categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)                 # (50000, 10)


#2. model
input1 = Input(shape = (32*32*3, ))

dense1 = Dense(100, activation = 'relu')(input1)
drop1 = Dropout(0.2)(dense1)

dense2 = Dense(50, activation = 'relu')(drop1)
drop2 = Dropout(0.2)(dense2)

dense2 = Dense(50, activation = 'relu')(drop2)
drop2 = Dropout(0.2)(dense2)

dense2 = Dense(50, activation = 'relu')(drop2)
drop2 = Dropout(0.2)(dense2)

dense3 = Dense(40, activation = 'relu')(drop2)
drop3 = Dropout(0.2)(dense3)

dense4 = Dense(35, activation = 'relu')(drop3)
dense5 = Dense(20,  activation = 'relu')(dense4)

output1 = Dense(10, activation = 'softmax')(dense5)

model = Model(inputs = input1, outputs = output1)

# ealrystopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode='auto', patience =50, verbose =1)


#3. fit
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs= 100, batch_size = 256, callbacks = [es],
              validation_split =0.2, shuffle = True, verbose =2)


#4. evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size = 256)
print('loss: ', loss)
print('acc: ', acc)
