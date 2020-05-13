#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1 ))
model.add(Dense(3))
# model.add(Dense(1000000))
# model.add(Dense(1000000)) : 두번째 부터 처리 안됌
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
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs =30, batch_size =2)

#4. 평가,예측
loss, acc= model.evaluate(x, y, batch_size =1) #평과 결과(손실, 정확도)를 loss, acc(변수)에 받겠다 
print("loss : ", loss)
print("acc : ", acc)
