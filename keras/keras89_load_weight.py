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
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
model = Sequential()
model.add(Conv2D(100, (2, 2), input_shape  = (28, 28, 1), padding = 'same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(80, (2, 2), padding = 'same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(60, (2, 2), padding = 'same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(40, (2, 2),padding = 'same'))
model.add(Conv2D(20, (2, 2),padding = 'same'))
model.add(Conv2D(10, (2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))                
# model.add(Dense(10, activation='softmax'))              
# ValueError: You are trying to load a weight file containing 7 layers into a model with 8 layers.           

model.summary()



# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 20, mode= 'auto')


#3. 훈련                      
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc']) # metrics=['accuracy']
'''
hist = model.fit(x_train, y_train, epochs= 10, batch_size= 64, callbacks = [es],
                                   validation_split=0.2, verbose = 1)
'''


""" load_weight """
model.load_weights('./model/test_weight1.h5')   # 각각의 레이어의 weight가 save된걸 가져온다 (model 불러오기 X)     
                                                # model구성, compile, fit부분이 필요 O  
                                                # weight가 저장된 모델과 구성이 동일해야 한다. 
                                                # : 저장된 weight수 만큼 node와 layer가 매칭되어야 하기 때문


#4. 평가
loss_acc = model.evaluate(x_test, y_test, batch_size= 64)
print('loss_acc: ', loss_acc)
   
# loss_acc:  [0.05392002098002813, 0.9842000007629395]

