#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

#2.모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

model = Sequential()
model.add(Dense(10, input_shape = (1, )))      # activation : default = 'linear'
model.add(Dense(100, activation = 'sigmoid'))  #            : 각 layer에서 계산된 결과값이 activation을 통해 다음 layer로 전달됌
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

#3.훈련
model.compile(loss = ['binary_crossentropy'], optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=1)

#4,평가, 예측

loss = model.evaluate(x_train, y_train)
print("loss: ", loss)       

x1_pred = np.array([11,12,13,14])

y_pred = model.predict(x1_pred)
print(y_pred)