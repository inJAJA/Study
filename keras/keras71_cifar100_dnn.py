import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt

#1. data
from keras.datasets import cifar100
(x_train, y_train),(x_test, y_test) = cifar100.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# x
x_train = x_train.reshape(x_train.shape[0], 32*32*3)
x_test = x_test.reshape(x_test.shape[0], 32*32*3)

# y 
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)


#2. model
input1 = Input(shape = (32*32*3, ))

dense1 = Dense(150, activation = 'relu')(input1)
dense2 = Dropout(0.2)(dense1)

dense1 = Dense(120, activation = 'relu')(dense2)
dense2 = Dropout(0.2)(dense1)

dense1 = Dense(100, activation = 'relu')(dense2)
dense2 = Dropout(0.2)(dense1)

dense1 = Dense(80, activation = 'relu')(dense2)
dense2 = Dropout(0.2)(dense1)

dense1 = Dense(60, activation = 'relu')(dense2)
dense2 = Dropout(0.2)(dense1)

output1 = Dense(100, activation = 'softmax')(dense2)

model = Model(inputs = input1, outputs = output1)

model.summary()


# callbacks
# EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience =50, verbose =1)
# checkpoint
modelfath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath = modelfath, monitor = 'val_loss',
                             save_best_only = True)
# Tensorboard
ts_board = TensorBoard(log_dir = 'graph',  histogram_freq = 0, 
                       write_graph = True, write_images = True)


#3. fit
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 64,
                 validation_split = 0.2, 
                 callbacks = [es, checkpoint, ts_board])

#4. evaluate
loss_acc = model.evaluate(x_test, y_test, batch_size = 64)
print('loss_acc: ' ,loss_acc)


#  matplotlib
plt.figure(figsize = (10, 6))

# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = 'o', c= 'green', label = 'loss')
plt.plot(hist.history['val_loss'], marker = 'o', c= 'red', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker = 'o', c= 'green', label = 'acc')
plt.plot(hist.history['val_acc'], marker = 'o', c= 'red', label = 'val_acc')
plt.grid()
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()

plt.show()