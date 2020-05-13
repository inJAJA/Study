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
# model.add(Dense(1000000)) : 두번째 부터 처리 안됌, cpu부족
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
# mse(Mean Squered Error,평균 제곱 오차) : sigma(실제값 - 예측값)^2 /n
# metrics 에 accuracy는 분류지표. loss의 지표와 metrics의 지표 방식이 서로 다름

model.fit(x, y, epochs =30, batch_size =1)

#4. 평가,예측
loss, acc= model.evaluate(x, y, batch_size =1) #평과 결과(손실, 정확도)를 loss, acc(변수)에 받겠다 
print("loss : ", loss)
print("acc : ", acc)

y_pred =model.predict(x_pred) # 예측 값을 y_pred에 반환
print("y_pred : ", y_pred)

#머신 예측방식
#1. 회귀 방식 : 우리가 수치를 넣었을 때 수치로 답을 해줌. ex) mse(평균 제곱 오차)
#2. 분류 방식 : 결과값에 대한 분류가 정해져 있어야 함. ex) 0과 1, 강아지와 고양이
#                                                      고양이 = 1
#                                                      강아지 = 0   ==> 0,1로 나옴(결과값으로 3이 나올수 없음)
#                                                      회귀 방식은 수치가 자유롭게 나옴으로 서로 중돌됨