import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist                         
mnist.load_data()                                         

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
print(x_train[0])                                         
print('y_train: ' , y_train[0])                           # 5

print(x_train.shape)                                      # (60000, 28, 28)
print(x_test.shape)                                       # (10000, 28, 28)
print(y_train.shape)                                      # (60000,)       
print(y_test.shape)                                       # (10000,)


# 데이터 전처리 1. 원핫인코딩 : 당연하다             
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                                      #  (60000, 10)

# 데이터 전처리 2. 정규화( MinMaxScalar )                                                    
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') /255  
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') /255.                                     



#2. 모델 구성
from keras.layers import Dense
from keras.models import load_model
model = load_model('./model/model_test01.h5')                  # fit한데까지만 저장됌
model.add(Dense(30, activation = 'relu', name = 'add1'))
model.add(Dense(20, activation = 'relu', name = 'add2'))
model.add(Dense(10, activation = 'softmax', name = 'add3'))


model.summary()



#4. 평가
loss_acc = model.evaluate(x_test, y_test, batch_size= 64)

print('loss_acc: ', loss_acc)                     

y_predict = model.predict(x_test[0: 10])
y_predict = np.argmax(y_predict, axis =1)
print(y_predict)


'''
loss_acc:  [2.287560285949707, 0.1023000031709671]
'''


