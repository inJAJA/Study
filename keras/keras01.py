import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1,activation='relu' ))


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs =500, batch_size=1)  # epochs : 훈련횟수, 
                                            # batch_size : 넘겨주는 데이터 샘플의 size, defalut값 = 32

loss, acc = model.evaluate(x, y, batch_size=1)

print("loss : ", loss)
print("acc : ", acc)
