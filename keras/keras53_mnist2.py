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

# 데이터 전처리 2. 정규화( MinMaxScalar )    => x 값                                           
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') /255  
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') /255.   # 뒤에 ' . '을 써도 된다.                                  
#             cnn 사용을 위한 4차원       # 타입 변환       # (x - min) / (max - min) : max =255, min = 0                                      
#                                         : minmax를 하면 소수점이 되기때문에 int형 -> float형으로 타입변환


#2. 모델 구성
# 0 ~ 9까지 씌여진 크기가 (28*28)인 손글씨 60000장을 0 ~ 9로 분류하겠다. ( CNN + 다중 분류)
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


#3. 훈련                      # 다중 분류
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc']) 
model.fit(x_train, y_train, epochs= 16, batch_size= 64, 
                 validation_split=0.2, verbose = 1)


#4. 평가
loss, acc = model.evaluate(x_test, y_test, batch_size= 64)
print('loss: ', loss)
print('acc: ', acc)



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