import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(5, input_dim =1))
model.add(Dense(10))
model.add(Dense(1))

model.compile( loss ='mse', optimizer='adam', metrics=['mse'])
model.fit(x, y, epochs =100, batch_size =1)

loss, mse = model.evaluate(x, y, batch_size = 1)

y_predict = model.predict(x)
print(y_predict)