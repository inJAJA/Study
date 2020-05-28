import numpy as np


#1. 데이터
from keras.datasets import mnist
mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)                              # (60000, 28, 28)
print(x_test.shape)                               # (10000, )
print(y_train.shape)                              # (60000, )
print(y_test.shape)                               # (10000, )


# x_data전처리 : MinMaxScaler
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


# y_data 전처리 : one_hot_encoding (다중 분류)
from keras.utils.np_utils import to_categorical
y_trian = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)


# reshape : Dense형 모델 사용을 위한 '2차원'
x_train = x_train.reshape(60000, 28*28 ) 
x_test = x_test.reshape(10000, 28*28)
print(x_train.shape)                              # (60000, 784)
print(x_test.shape)                               # (10000, 784)



#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(200, activation = 'relu',input_dim = 28*28))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))                                             # Dropout 사용
model.add(Dense(80, activation = 'relu'))
model.add(Dropout(0.2))                                             # Dropout 사용
model.add(Dense(60, activation = 'relu'))
model.add(Dropout(0.2))                                             # Dropout 사용
model.add(Dense(40, activation = 'relu'))
model.add(Dropout(0.2))                                             # Dropout 사용
model.add(Dense(20, activation = 'relu'))
model.add(Dropout(0.2))                                             # Dropout 사용
model.add(Dense(10, activation = 'softmax'))

model.summary()

# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 50, verbose = 1 )


#3. 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'] )
model.fit(x_train, y_trian, epochs = 100, batch_size = 64, 
                            validation_split =0.2,
                            shuffle = True, callbacks = [es])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size= 64)
print('loss: ', loss)
print('acc: ', acc)

# acc:  0.9785000085830688