'''
# minmax_scalar   ( 정규화 )  =  ( x - min ) / ( max - min )
  : 0 ~ 1 사이의 값으로 변환   

# standard_scalar ( 표준화 )  =  ( x - x평균) / x표준편차      
  : 0을 기준으로 모임 (정규분포 모양)

# 표준편차 = [ sigma ( x - x평균)^2 ] / n        
'''
from numpy import array
from keras.models import Model
from keras.layers import Dense, LSTM, Input

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],            
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[11,12,13],
           [2000,3000,4000],[3000,4000,5000],[4000,5000,6000],   # (14, 3) 
           [100, 200, 300]
          ])
y = array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000, 400])                      # (14, )   벡터

x_predict = array([55, 65, 75])           # (3, )

print('x.shape : ',x.shape)               # (14, 3)
print('y.shape : ',y.shape)               # (14, ) != (14, 1)
                                          #  벡터      행렬
x_predict = x_predict.reshape(1, 3)       # (1, 3, ) 도 가능하다 -> , 뒤에 값이 없어서

##### MinMax_scaler , Standard_scaler ###### 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x)                             # 전처리 실행 : x를 넣어서 MinMax_scaler, Standard_scaler 실행하겠다.
                                          #              scaler에 실행한 값 저장 ( x의 범위 )
x = scaler.transform(x)                   # x의 모양을 MinMaxScaler을 실행한 값으로 바꿔주겠다.
x_predict = scaler.transform(x_predict)   # x의 범위로 계산한 sclar 값에서 x_predict에 해당되는 값을 가져오겠다.
print(x)
print(x_predict)

""" y는 전처리 변환을 안하는 이유 : x와 매칭되는 순서는 같기 때문
ec)      x1  x2    x3    x4  ... x99   x100
     ---------------------------------------
     x = 1  2     3     4    ... 99    100
scalar = 0  0.01  0.02  0.02 ... 0.99  1
    ========================================
     y = y1  y2    y3    y4  ... y99   y100
"""
                                          


# x = x.reshape(14, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)  # x.shape[0] = 14 / x.shape[1] = 3 / data 1개씩 작업 하겠다. 
print(x.shape)                            # (14, 3, 1)    


#2. 모델구성

input1 = Input(shape = (3, 1))

LSTM1 = LSTM(1000, return_sequences= True)(input1)
# LSTM2 = LSTM(10)(LSTM1, return_sequences= True)(LSTM1)  # return_sequences를 썼으면 무조건 LSTM사용
LSTM2 = LSTM(500)(LSTM1)           
dense1 = Dense(10)(LSTM2)        
dense2 = Dense(10)(dense1)     
dense2 = Dense(10)(dense2)   
dense2 = Dense(10)(dense2)                     
dense2 = Dense(10)(dense2)                     
dense2 = Dense(10)(dense2)                     
dense3 = Dense(10)(dense2)                     


output1 = Dense(1)(dense3)

model = Model(inputs = input1, outputs = output1)

model.summary()



# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=100, mode = 'min')


#3. 실행
model.compile(optimizer='adam', loss = 'mse')
model.fit(x, y, epochs =10000, batch_size = 32,callbacks = [es] 
          )                

#4. 예측
x_predict = x_predict.reshape(1, 3, 1)   # x값 (14, 3, 1)와 동일한 shape로 만들어 주기 위함
                                         # (1, 3, 1) : 확인 1 * 3 * 1 = 3
# x_predict = x_predict.reshape(1, x_predict.shape[0], 1)

print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)
