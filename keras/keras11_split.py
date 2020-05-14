#1. 데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(101, 201))

x_train = x[:60]  # python시퀀스 자료형 슬라이스 참조
x_val = x[60:80]
x_test = x[80:]

y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

print(x_train)
print(x_val )
print(x_test )


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1 ))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(70))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit(x_train, y_train, epochs =100, batch_size =1,
          validation_data = (x_val, y_val))

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

 