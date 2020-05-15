#1. 데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(101, 201))  # train, test, validation 나누기(직접 손으로)

'''
# range(a, b) : a에서 (b-1)까지 생성됨
#             : a값 설정 안할 시 = 0
# 만들어진 후 0부터 시작되는 순서 부여됨 
'''

x_train = x[:60]  # python시퀀스 자료형 슬라이스 참조
x_val = x[60:80]
x_test = x[80:]

'''
# 실제 : 1 2 3 4 5 6 7 8 ~ 58 59 60 61 62 ~ 78 79 80 81 82 ~ 98 99 100
# 배열 : 0 1 2 3 4 5 6 7 ~ 57 58 59 60 61 ~ 77 78 79 80 81 ~ 97 98  99   

# x[:60] = 1 ~ 60
#   배열 : 0 ~ 59 까지 출력
# 배열의 순서로 따짐 : 0부터 (60 - 1)번 째 까지 

# x[60:80] = 61 ~ 80
#     배열 : 60 ~ 79 (80 - 1)            

# x[80:] = 81 ~ 100
#   배열 : 80 ~ 99
'''

y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

print(x_train)
print(x_val )
print(x_test )

'''
#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1 ))
model.add(Dense(20))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(230))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(40))
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
'''
