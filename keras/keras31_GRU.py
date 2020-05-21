from numpy import array
from keras.models import Sequential
from keras.layers import Dense, GRU

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7])                      # (4, )   벡터


print('x.shape : ',x.shape)               # (4, 3)
print('y.shape : ',y.shape)               # (4, ) != (4, 1)
                                          #  벡터      행렬

# x = x.reshape(4, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)  # x.shape[0] = 4 / x.shape[1] = 3 / data 1개씩 작업 하겠다. 
print(x.shape)                            # (4, 3, 1)      / rehape확인 : 모든 값을 곱해서 맞으면 똑같은 거임


#2. 모델구성
model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape = (3, 1)))
model.add(GRU(700, input_length =3, input_dim= 1))                # input_length : time_step (열)
model.add(Dense(500))   
model.add(Dense(400)) 
model.add(Dense(300))   
model.add(Dense(200))   
model.add(Dense(100)) 
model.add(Dense(100))  # 100 : 6  


model.add(Dense(90))   
model.add(Dense(50))   
model.add(Dense(10))   
model.add(Dense(1))


model.summary()
'''
GRU_parameter 계산
num_params = 3 * ( num_units   +   input_dim   +   1 )  *  num_units
                (output node값)  (잘라준 data)   (bias)  (output node값)
           = 3 * (    5        +       1       +   1 )  *     5          = 105     
                    역전파 : 나온 '출력' 값이 다시 '입력'으로 들어감(자기회귀)
'''

# # EarlyStopping
# from keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor = 'loss', patience=100, mode = 'min')


#3. 실행
model.compile(optimizer='adam', loss = 'mse')
model.fit(x, y, epochs =1500, batch_size = 32 #,callbacks =[es]
          )                

#4. 예측
x_predict = array([5, 6, 7])               # (3, )
x_predict = x_predict.reshape(1, 3, 1)     # x값 (4, 3, 1)와 동일한 shape로 만들어 주기 위함
                                         # (1, 3, 1) : 확인 1 * 3 * 1 = 3
# x_predict = x_predict.reshape(1, x_predict.shape[0], 1)

print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)
