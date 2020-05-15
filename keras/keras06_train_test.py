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
loss, mse= model.evaluate(x_test, y_test, batch_size =1) 
'''
# 한가지 데이터로만 학습을 계속하면 과적합(Overfitting)이 발생하여 일반화의 성능이 떨이짐.
# 새로운 값(x_test)을 주어 비교하여 일반화 성능을 높임, 완전 새로운 값(x_predict)을 넣었을 때 좀 더 정확한 예측값이 나온다(y_predict)
# 평가 데이터는 모델에 반영 안된다. 그래서 통상적으로 [train data > test data]
'''

print("loss : ", loss)
print("mse : ", mse)

y_pred =model.predict(x_pred) 
print("y_pred : ", y_pred)