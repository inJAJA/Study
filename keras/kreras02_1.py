from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([5,10,15,20,25,30,35,40,45,50])
x_test = np.array([21,22,23,24,25,26,27,28,29,30])
y_test = np.array([105,110,115,120,125,130,135,140,145,150])

model = Sequential()
model.add(Dense(5, input_dim =1, activation='relu'))
model.add(Dense(20))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1, activation='relu'))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs =1000, batch_size=100, validation_data = (x_train, y_train))
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

print("loss : ", loss)
print("acc : ", acc)

output = model.predict(x_test)
print(output)