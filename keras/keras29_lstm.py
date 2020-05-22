from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7])                      # (4, )   벡터
y2 = array([[4,5,6,7]])                   # (1, 4)  행렬
y3 = array([[4],[5],[6],[7]])             # (4, 1)

print('x.shape : ',x.shape)               # (4, 3)
print('y.shape : ',y.shape)               # (4, ) != (4, 1)
                                          #  벡터      행렬

# x = x.reshape(4, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)  # x.shape[0] = 4 / x.shape[1] = 3 / data 1개씩 작업 하겠다. 
print(x.shape)                            # (4, 3, 1)      / rehape확인 : 모든 값을 곱해서 맞으면 똑같은 거임


#2. 모델구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape = (3, 1))) # LSTM설정 : ( 열, 몇개씩 짤라서 작업할 것인가)
model.add(Dense(11))                       # Hidden layer
model.add(Dense(17))    
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(5))   
model.add(Dense(1))


model.summary()
'''
# x.shape = (4, 3, 1)

Layer (type)                 Output Shape              Param #
=================================================================
gru_1 (GRU)                  (None, 10)                  480
_________________________________________________________________
dense_1 (Dense)              (None, 10)                  110
_________________________________________________________________
dense_2 (Dense)              (None, 3)                   33 
_________________________________________________________________
dense_3 (Dense)              (None, 1)                   4
=================================================================

# LSTM_parameter 계산
 : num_params = 4 * ( num_units   +   input_dim   +   1 )  *  num_units
                   (output node값)  (잘라준 data)   (bias)  (output node값)
              = 4 * (    10       +       1       +   1 )  *     10          = 480  
                       역전파 : 나온 '출력' 값이 다시 '입력'으로 들어감(자기회귀)   
                4 : LSTM_gate연산 ( forget, input, output_gate / cell_state)
'''


# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=50, mode = 'min')

#3. 실행
model.compile(optimizer='adam', loss = 'mse')
model.fit(x, y, epochs =1000, batch_size = 1, callbacks = [es] )               


#4. 예측
x_predict = array([5, 6, 7])               # (3, )
x_predict = x_predict.reshape(1, 3, 1)     # x값 (4, 3, 1)와 동일한 shape로 만들어 주기 위함
                                         # (1, 3, 1) : 확인 1 * 3 * 1 = 3
# x_predict = x_predict.reshape(1, x_predict.shape[0], 1)

print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)
