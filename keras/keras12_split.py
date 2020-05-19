#1. 데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(101, 201))

# train_test_split(sklearn)사용하여  train, test, validation 나누기

from sklearn.model_selection import train_test_split # sklearn안에 model안에 
x_train, x_test, y_train, y_test = train_test_split( 
    # x, y, random_state=66, shuffle = True, # shuffle의 defalut = True
    # random_state(섞는 방식,순서)가 없으면 섞을 때마다 매번 나오는 값이 다름 -> 매번 미세하게 결과가 달라짐
    x, y, shuffle = False,
    train_size =0.6
    )
    # x, y의 전체 데이터 받기, train_size를 전체 데이터의 60%를 받겠다.

'''
# #shuffle을 하는 이유?
# : train와 test data의 범위가 완전히 분리되어 있으면 test값(train범위 외의 구간)을 제대로 유추 못할 수 있다.
#   ( 한번도 Train에서 경험하지 못했기 때문에)
# : 그래서 train과 test data범위가 겹치는 것이 정확도를 올리는데 좋음
#
# #shuffle 조건
# : x, y를 쌍으로 넣어야 함 -> x와 y가 매칭되어야 하기 때문에
'''

x_val, x_test, y_val, y_test = train_test_split( 
    x_test, y_test, random_state=66,
    # x_test, y_test, shuffle = False,
    test_size =0.5
    )
# x_test, y_test 데이터 받아서 그 데이터 50%를 test_size로 설정 

# x_train = x[:60]  # python시퀀스 자료형 슬라이스 참조
# x_val = x[60:80]
# x_test = x[80:]

# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]

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
model.fit(x_train, y_train, epochs = 70, batch_size =1,
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

 