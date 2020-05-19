#1. 데이터
import numpy as np
x = np.transpose([range(1, 101), range(311, 411), range(100)])  # python의 list
# x = np.array([range(1, 101), range(311, 411), range(100)]).T
# x = np.swapaxes([range(1, 101), range(311, 411), range(100)], 0, 1)
# x = np.array([range(1, 101), range(311, 411), range(100)]).reshape(100, 3)
y = np.transpose([range(101, 201), range(711, 811), range(100)])

print(x.shape)  # np.array() = (3, 100) - 100개의 column(data종류)에 data 3개가 들어감
'''
# ['열(column)' 우선, "행" 무시]
# column(열) : data의 종류 = input_dim에 들어가는 갯수 ex)날씨, 돈, 주가 등등
# 행 : column에 들어가는 data의 갯수

# 행, 열 바꾸는 법
# 1. np.swapaxes( , 0, 1) : 만들어지는 것의 행과 열을 반전 ex) (a, b) -> (b, a) 
# 2. np.transpos    : 동일
# 3. .T             : 동일         
# 4. .reshape(a, b) : a행 b열의 모습으로 다시 만들어줌 ex) (c, d) -> (a, b)
'''

from sklearn.model_selection import train_test_split     # 행(data 개수)으로 짤려서 열에 들어감
x_train, x_test, y_train, y_test = train_test_split(  
    # x, y, random_state=66, shuffle = True,
    x, y, shuffle = False,
    train_size =0.8                                      # (80, 3)
    )

# x_val, x_test, y_val, y_test = train_test_split( 
#     # x_test, y_test, random_state=66,
#     x_test, y_test, shuffle = False,
#     test_size =0.5
#     )    

# x_train = x[:60]  # python시퀀스 자료형 슬라이스 참조
# x_val = x[60:80]
# x_test = x[80:]

# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]

print(x_train)
print(x_test )


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(5, input_dim = 3 ))                   # input layer 
model.add(Dense(10))
model.add(Dense(17))
model.add(Dense(3))                                   # output layer


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit(x_train, y_train, epochs =100, batch_size =1,
        # validation_data = (x_val, y_val)
          validation_split= 0.25                         
          )                                                


#4. 평가,예측
loss, mse = model.evaluate(x_test, y_test, batch_size =1) 
print("loss : ", loss)
print("mse : ", mse)

# y_pred = model.predict(x_pred)  #눈으로 보기 위한 예측값
# print("y_pred : ", y_pred)

y_predict = model.predict(x_test)  
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


model.summary()