## 앙상블 모델을 만드시오

from numpy import array
from keras.models import Model
from keras.layers import Dense, LSTM, Input

# 1. 데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],      # (13, 3)
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[11,12,13],
           [20,30,40],[30,40,50],[40,50,60]
          ])
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],  # ( 13, 3)
           [50,60,70],[60,70,80],[70,80,90],[80,90,100],
           [90,100,110],[110,120,130],
           [2,3,4],[3,4,5],[4,5,6]
          ])          
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])                      # (13, )   벡터

x1_predict = array([55, 65, 75])           # (3, )
x2_predict = array([65, 75, 85])           # (3, )


print('x.shape : ',x1.shape)              # (13, 3)
print('x.shape : ',x2.shape)              # (13, 3)
print('y.shape : ',y.shape)               # (13, ) != (13, 1)
                                          #  벡터      행렬

# x = x.reshape(13, 3, 1)
x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)  # x.shape[0] = 13 / x.shape[1] = 3 / data 1개씩 작업 하겠다. 
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)  
print(x1.shape)                               # ( 13, 3, 1) 
print(x2.shape)                               # ( 13, 3, 1) 


#2. 모델구성

# model = Sequential()
# # model.add(LSTM(10, activation='relu', input_shape = (3, 1)))
# model.add(LSTM(700, input_length =3, input_dim= 1))                # input_length : time_step (열)
# model.add(Dense(101))   
# model.add(Dense(100))   
# model.add(Dense(100))   
# model.add(Dense(100))   
# model.add(Dense(100)) 
# model.add(Dense(90)) 
# model.add(Dense(51))   # 5 
# model.add(Dense(1))

# 모델 1
input1 = Input(shape = (3, 1))
input1_2 = LSTM(10)(input1)
input1_3 = Dense(5)(input1_2)

# 모델 2
input2 = Input(shape = (3, 1))
input2_2 = LSTM(10)(input2)
input2_3 = Dense(5)(input2_2)

# 병합
from keras.layers.merge import concatenate
merge1 = concatenate([input1_3, input2_3])

middle1 = Dense(10)(merge1)
middle2 = Dense(10)(middle1)
middle3 = Dense(10)(middle2)

# output
output1 = Dense(5)(middle3)
output1_2 = Dense(5)(output1)
output1_3 = Dense(1)(output1_2)

model = Model(inputs = [input1, input2], outputs = output1_3)

model.summary()



# # EarlyStopping
# from keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor = 'loss', patience=100, mode = 'min')

#3. 실행
model.compile(optimizer='adam', loss = 'mse')
model.fit([x1, x2], y, epochs =800, batch_size = 32 #,callbacks = [es] 
          )                

#4. 예측
x1_predict = x1_predict.reshape(1, 3, 1)   # x값 (13, 3, 1)와 동일한 shape로 만들어 주기 위함
                                           # (1, 3, 1) : 확인 1 * 3 * 1 = 3
# x_predict = x_predict.reshape(1, x_predict.shape[0], 1)
x2_predict = x2_predict.reshape(1, 3, 1)

# print(x1_predict)

y_predict = model.predict([x1_predict, x2_predict])
print(y_predict)
