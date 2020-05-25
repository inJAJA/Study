#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
# x_pred = np.array([16, 17, 18])
  
x_val = np.array([101,102,103,104,105])                       # validation형성
y_val = np.array([101,102,103,104,105])
'''
validation_data
: 모델의 fit부분에서 사용되어 진다.
  모델을 x_train에서 훈련을 시키고 x_val을 가지고 검증한다.
: x_val  : 컴퓨터용 검증 -> model에 적용 O
  x_test : 사람용 평가   -> model에 적용 X
: data양 : x_train > x_val
'''

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1 ))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit(x_train, y_train, epochs =100, batch_size =1,
          validation_data = (x_val, y_val))                    # validation_data = ( , )

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



 