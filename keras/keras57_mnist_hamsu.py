import numpy as np

from keras.datasets import mnist                          # keras에서 제공되는 예제 파일 

mnist.load_data()                                         # mnist파일 불러오기

(x_train, y_train), (x_test, y_test) = mnist.load_data()  

print(x_train[0])                                         # 0 ~ 255까지의 숫자가 적혀짐 (color에 대한 수치)
print('y_train: ' , y_train[0])                           # 5

print(x_train.shape)                                      # (60000, 28, 28)
print(x_test.shape)                                       # (10000, 28, 28)
print(y_train.shape)                                      # (60000,)        : 10000개의 xcalar를 가진 vector(1차원)
print(y_test.shape)                                       # (10000,)



# 데이터 전처리 1. 원핫인코딩 : 당연하다             
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                                      #  (60000, 10)

# 데티어 전처리 2. 정규화( MinMaxScalar )                                              
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') /255  
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') /255.                                    



from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from keras.layers import Dropout                   

input1 = Input(shape = (28, 28, 1) )
dense1 = Conv2D(200, (3, 3), padding = 'same')(input1)
maxpool1 = MaxPooling2D(pool_size=2)(input1)
drop1 = Dropout(0.2)(maxpool1)                               # Dropout 사용

dense2 = Conv2D(100, (2, 2), padding = 'same')(drop1)
maxpool2 = MaxPooling2D(pool_size=2)(dense2)
drop2 = Dropout(0.3)(maxpool2)                               # Dropout 사용

dense3 = Conv2D(80, (2, 2), padding = 'same')(drop2)
maxpool3 = MaxPooling2D(pool_size=2)(dense3)
drop3 =Dropout(0.3)(maxpool3)                                # Dropout 사용

dense4 = Conv2D(60, (2, 2),padding = 'same')(drop3)
drop4 = Dropout(0.3)(dense4)                                 # Dropout 사용

dense5 = Conv2D(40, (2, 2),padding = 'same')(drop4)
drop5 = Dropout(0.3)(dense5)                                 # Dropout 사용

dense6 = Conv2D(20, (2, 2), padding='same')(drop5)
flat = Flatten()(dense6)
output1 = Dense(10, activation='softmax')(flat)              

model = Model(inputs = input1 , outputs = output1)

model.summary()

# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience = 50, mode = 'auto', verbose = 1)

#3. 훈련                     
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc']) 
model.fit(x_train, y_train, epochs= 100, batch_size= 64, verbose = 2,
                 validation_split=0.2,
                 callbacks = [es] )



#4. 평가
loss, acc = model.evaluate(x_test, y_test, batch_size= 64)
print('loss: ', loss)
print('acc: ', acc)


