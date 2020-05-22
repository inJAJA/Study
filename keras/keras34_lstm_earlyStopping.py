## 함수형으로 리뉴얼하시오.

from numpy import array
from keras.models import Model
from keras.layers import Dense, LSTM, Input

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],            
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[11,12,13],
           [20,30,40],[30,40,50],[40,50,60],
          ])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])                      # (13, )   벡터

x_predict = array([50, 60, 70])           # (3, )

print('x.shape : ',x.shape)               # (13, 3)
print('y.shape : ',y.shape)               # (13, ) != (13, 1)
                                          #  벡터      행렬

# x = x.reshape(13, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)  # x.shape[0] = 13 / x.shape[1] = 3 / data 1개씩 작업 하겠다. 
print(x.shape)                            # (13, 3, 1)    


#2. 모델구성

input1 = Input(shape = (3, 1))

dense1 = LSTM(10)(input1)
dense2 = Dense(500)(dense1)
dense2 = Dense(200)(dense1)
dense2 = Dense(100)(dense1)
dense2 = Dense(80)(dense1)
dense2 = Dense(50)(dense1)
dense2 = Dense(20)(dense1)
dense3 = Dense(15)(dense2)

output1 = Dense(1)(dense3)

model = Model(inputs = input1, outputs = output1)

model.summary()


# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=100, mode = 'min')


#3. 실행
model.compile(optimizer='adam', loss = 'mse')
model.fit(x, y, epochs =10000, batch_size = 13,callbacks = [es] 
          )                

#4. 예측
x_predict = x_predict.reshape(1, 3, 1)   # x값 (13, 3, 1)와 동일한 shape로 만들어 주기 위함
                                         # (1, 3, 1) : 확인 1 * 3 * 1 = 3
# x_predict = x_predict.reshape(1, x_predict.shape[0], 1)

print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)
