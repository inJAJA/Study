from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7])                      # (4, )   벡터


print('x.shape : ',x.shape)               # (4, 3)
print('y.shape : ',y.shape)               # (4, ) != (4, 1)
                                          #  벡터      행렬

# x = x.reshape(4, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)  # x.shape[0] = 4 / x.shape[1] = 3 / data 1개씩 작업 하겠다. 
print(x.shape)                            # (4, 3, 1)      / rehape확인 : 모든 값을 곱해서 맞으면 똑같은 거임
'''
                행            열        몇개씩 자르는 지
x.shape = ( batch_size , time_steps , feature )

input_shape = (time_steps, feature )

input_length = timesteps
input_dim = feature

                 x      | y
            ---------------- 
batch_size   1   2   3  | 4     : x의 한 행에 들어간 값을 몇개씩 자르느냐 = feature
             2   3   4  | 5       ex) feature 1 : [1], [2], [3]
             3   4   5  | 6       ex) feature 3 : [1, 2, 3]
             4   5   6  | 7 
              time_step

'''

#2. 모델구성
model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape = (3, 1)))
model.add(LSTM(5, input_length =3, input_dim= 1))                # input_length : time_step (열)
model.add(Dense(3))   
model.add(Dense(1))


model.summary()
'''
LSTM_parameter 계산
num_params = 4 * ( num_units   +   input_dim   +   1 )  *  num_units
                (output node값)  (잘라준 data)   (bias)  (output node값)
           = 4 * (    5      +       1       +   1 )  *     5          = 140     
                    역전파 : 나온 '출력' 값이 다시 '입력'으로 들어감(자귀회귀)
'''


# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=100, mode = 'min')

#3. 실행
model.compile(optimizer='adam', loss = 'mse')
model.fit(x, y, epochs =2000, batch_size = 1, callbacks = [es] )                

#4. 예측
x_predict = array([5, 6, 7])               # (3, )
x_predict = x_predict.reshape(1, 3, 1)       # x값 (4, 3, 1)와 동일한 shape로 만들어 주기 위함
                                         # (1, 3, 1) : 확인 1 * 3 * 1 = 3
# x_predict = x_predict.reshape(1, x_predict.shape[0], 1)

print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)
