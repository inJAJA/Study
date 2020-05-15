#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16, 17, 18])


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1 ))
model.add(Dense(10))
# model.add(Dense(1000000))
# model.add(Dense(1000000)) 
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(20))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(5))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit(x_train, y_train, epochs =100, batch_size =1)

#4. 평가,예측
loss, mse = model.evaluate(x_test, y_test, batch_size =1) 
# 평가시 mse 식의 예측값이 생겨서 mse를 구할 수 있음
print("loss : ", loss)
print("mse : ", mse)

# y_pred = model.predict(x_pred)  #눈으로 보기 위한 예측값
# print("y_pred : ", y_pred)

y_predict = model.predict(x_test)  # RMSE의 계산을 위한  x_test의 예측값
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error  # tensorflow, keras가 나오기 전에 쓰던 API
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))   # sprt() : root를 씌어줌
print("RMSE : ", RMSE(y_test, y_predict))

'''
# RMSE (Root Mean Squared Error, 평균 제곱근 오차) 
# : root[sigma(실제값 - 예측값)^2 /n]
# : MSE에 비해 에러값의 크기가 실제 값에 비례한다.(제곱된 것에 루트를 씌었기 때문)
'''