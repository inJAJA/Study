# keras56_mnist_DNN.py 땡겨라.
# input_dim =154로 모델을 만드시오.

import numpy as np


#1. 데이터
from keras.datasets import mnist
mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)                              # (60000, 28, 28)
print(x_test.shape)                               # (10000, )
print(y_train.shape)                              # (60000, )
print(y_test.shape)                               # (10000, )


# y_data 전처리 : one_hot_encoding (다중 분류)
from keras.utils.np_utils import to_categorical
y_trian = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)


x_train = x_train.reshape(60000, 28*28 ).astype('float32')/255 
x_test = x_test.reshape(10000, 28*28).astype('float32')/255
print(x_train.shape)                              # (60000, 784)
print(x_test.shape)                               # (10000, 784)

X = np.append(x_train, x_test, axis = 0)

from sklearn.decomposition import PCA
pca = PCA(n_components = 154)
# x_train = pca.fit_transform(x_train)    # x_test에 대한 특성이 배제됌
# x_test = pca.transform(x_test)
X = pca.fit_transform(X)

x_train = X[:60000]
x_test = X[60000:]
print(x_train.shape)                              # (60000, 154)
print(x_test.shape)                               # (10000, 154)


#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(200, activation = 'relu',input_dim = 154))
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
# acc:  0.9763000011444092 -> pca후