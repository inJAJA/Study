import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, Input
from keras.models import load_model
from keras.callbacks import EarlyStopping


x_train, x_test, y_train, y_test = np.load('./img_data.npy', allow_pickle = True)

print(x_train.shape)      # (130, 50, 100, 3)
print(y_train.shape)      # (130, 9)


model = Sequential()
model.add(Conv2D(50, (2, 2), input_shape = (50, 100, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size= 2))
model.add(Dropout(0.2))
model.add(Conv2D(100, (2, 2), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size= 2))
model.add(Dropout(0.2))
model.add(Conv2D(100, (2, 2), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size= 2))
model.add(Dropout(0.2))
model.add(Conv2D(100, (2, 2), padding = 'same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size= 2))
model.add(Dropout(0.2))
model.add(Conv2D(100, (2, 2), padding = 'same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size= 2))
model.add(Dropout(0.2))
model.add(Conv2D(100, (2, 2), padding = 'same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size= 2))
model.add(Dropout(0.2))
model.add(Conv2D(100, (2, 2), padding = 'same', activation = 'relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(9, activation = 'softmax'))

es = EarlyStopping(monitor = 'val_loss', patience = 100)

model.compile( loss = 'categorical_crossentropy', metrics = ['acc'], optimizer = 'adam')
model.fit(x_train, y_train, epochs = 500, batch_size = 64, validation_split = 0.2,
         callbacks = [es])

model.save('./train_model.h5')

loss_acc =  model.evaluate(x_test, y_test, batch_size = 64)
print('loss_acc: ', loss_acc)


