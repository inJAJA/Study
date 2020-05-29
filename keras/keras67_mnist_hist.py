import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist                          # keras에서 제공되는 예제 파일 

mnist.load_data()                                         # mnist파일 불러오기

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # mnist에서 이미 x_train, y_train으로 나눠져 있는 값 가져오기

print(x_train[0])                                         # 0 ~ 255까지의 숫자가 적혀짐 (color에 대한 수치)
print('y_train: ' , y_train[0])                           # 5

print(x_train.shape)                                      # (60000, 28, 28)
print(x_test.shape)                                       # (10000, 28, 28)
print(y_train.shape)                                      # (60000,)        : 10000개의 xcalar를 가진 vector(1차원)
print(y_test.shape)                                       # (10000,)



# 데이터 전처리 1. 원핫인코딩 : 당연하다              => y 값  
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                                      #  (60000, 10)

# 데이터 전처리 2. 정규화( MinMaxScalar )            => x 값                                           
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
model.add(Dense(10, activation='softmax'))                # 다중 분류

model.summary()


#3. 훈련                      
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc']) # metrics=['accuracy']

# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 20, mode= 'auto')

hist = model.fit(x_train, y_train, epochs= 16, batch_size= 64, callbacks = [es],
                                   validation_split=0.2, verbose = 1)
# hist값이 epoch순으로 저장된다.


#4. 평가
loss_acc = model.evaluate(x_test, y_test, batch_size= 64)

loss = hist.history['loss']                       # model.fit 에서 나온 값
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('acc: ', acc)                               
print('val_acc: ', val_acc)
print('loss_acc: ', loss_acc)                     

import matplotlib.pyplot as plt    

plt.figure(figsize = (10, 6))                     # 10 x 6인치의 판이 생김

# 1번 그림
plt.subplot(2, 1, 1)                              # (2, 1, 1) 2행 1열의 그림 1번째꺼 / subplot : 2장 그림               
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')                     
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')                  
plt.grid()                                        # 격자 생성
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['loss','val_loss']) 
plt.legend(loc = 'upper right')                   # legend의 위치(location) 설정/ default = 제이 비어 있는 곳
                                                  # 위에 label이 지정되어서 안써도 된다.

# 2번 그림
plt.subplot(2, 1, 2)                              # (2, 1, 2) 2행 1열의 그림 2번째꺼               
plt.plot(hist.history['acc'])                     
plt.plot(hist.history['val_acc'])                  
plt.grid()                                        # 격자 생성
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc','val_acc'])

plt.show()                                         

'''
#자습 : x_test를 10행 가져와서 x_predict로 써보기
x_pred = x_test[:10]
print(x_pred.shape)                                       # (10, 28, 28, 1)
y_pred = y_test[:10]

y1_pred = np.argmax(y_test[:10], axis=1)                  # x_predict값에 매칭되는 실제 y_predict값
print('실제값: ',y1_pred)                                 # 실제값:  [7 2 1 0 4 1 4 9 5 9]

y2_pred = model.predict(x_pred)                           # x_predict값을 가지고 예측한 y_predict값
y2_pred = np.argmax(y2_pred, axis =1)
print('예측값: ', y2_pred)                                # 예측값:  [7 2 1 0 4 1 4 9 5 9]       

# acc:  0.98580002784729
'''