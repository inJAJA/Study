#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x_pred = np.array([11, 12, 13])
# predict

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1 ))
model.add(Dense(3))
# model.add(Dense(1000000))
# model.add(Dense(1000000)) 
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  # loss와 metrics에 동일하게 mse가 들어가서 같은 값 나옴

model.fit(x, y, epochs =30, batch_size =1)

#4. 평가,예측
loss, mse= model.evaluate(x, y, batch_size =1) 
# 잘못된 점 : 훈련시킨 데이터로 평가함, 이렇게 하면 accuracy가 높아지거나 loss가 낮아짐 
#                                   (이미 아는 답으로 문제를 풀기 때문, 과적화overfitting) 

print("loss : ", loss)
print("mse : ", mse)

y_pred =model.predict(x_pred) 
print("y_pred : ", y_pred)