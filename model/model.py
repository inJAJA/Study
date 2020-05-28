from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D, Dense, Flatten

#모델구성

model = Sequential()
model.add(Conv2D(20(4,4), input_shape = (20,20,1)))
model.add(Conv2D(10,3,3))
model.add(Conv2D(10,3,3))
model.add(Conv2D(10,3,3))
model.add(Conv2D(10,3,3))
model.add(Flatten())
model.add(Dense(1))

model.summary()