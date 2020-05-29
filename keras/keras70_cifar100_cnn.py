# cifar10 색상이 들어가 있다.
from keras.datasets import cifar100
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout, Input
import matplotlib.pyplot as plt

#1. data
(x_train, y_train),(x_test, y_test) = cifar100.load_data()

print(x_train[0])
print('y_train[0] :',y_train[0])

print(x_train.shape)                # (50000, 32, 32, 3)
print(x_test.shape)                 # (10000, 32, 32, 3)
print(y_train.shape)                # (50000, 1)
print(y_test.shape)                 # (10000, 1)


# plt.imshow(x_train[3])
# plt.show()

# y
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)                # (50000, 100) 

#2. model
input1 = Input(shape = (32, 32, 3))

dense1 = Conv2D(150, (2, 2), activation = 'relu', padding = 'same')(input1)
dense2 = MaxPooling2D(pool_size = 2)(dense1)
dense3 = Dropout(0.2)(dense2)

dense1 = Conv2D(100, (2, 2), activation = 'relu', padding = 'same')(dense3)
dense2 = MaxPooling2D(pool_size = 2)(dense1)
dense3 = Dropout(0.2)(dense2)

dense1 = Conv2D(60, (2, 2), activation = 'relu', padding = 'same')(dense3)
dense2 = MaxPooling2D(pool_size = 2)(dense1)
dense3 = Dropout(0.2)(dense2)

dense1 = Conv2D(40, (2, 2), activation = 'relu', padding = 'same')(dense3)
dense2 = MaxPooling2D(pool_size = 2)(dense1)
dense3 = Dropout(0.2)(dense2)

dense1 = Conv2D(20, (2, 2), activation = 'relu', padding = 'same')(dense3)
dense2 = MaxPooling2D(pool_size = 2)(dense1)
dense3 = Dropout(0.2)(dense2)

dense1 = Conv2D(10, (2, 2), activation = 'relu', padding = 'same')(dense3)
dense3 = Dropout(0.2)(dense1)

dense1 = Conv2D(10, (2, 2), activation = 'relu', padding = 'same')(dense3)
dense3 = Dropout(0.2)(dense1)

flat = Flatten()(dense3)
output1 = Dense(100, activation = 'softmax')(flat)

model = Model(inputs = input1, outputs = output1)

model.summary()


# callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
# EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience = 50, verbose =1)
# Checkpoint
modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                            save_best_only = True)
# tensorboard
board = TensorBoard(log_dir = 'graph', histogram_freq = 0,
                   write_graph = True, write_images = True)


#3. compile, fit
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 256,
                 validation_split =0.2 ,shuffle = True,
                 callbacks = [es, checkpoint, board])


#4. evaluate
loss_acc = model.evaluate(x_test, y_test, batch_size =256)
print('loss_acc: ' ,loss_acc)

# matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize = (10, 6))

# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '^', c = 'magenta', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '^', c = 'cyan', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker = '^', c = 'magenta', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '^', c = 'cyan', label = 'val_acc')
plt.grid()
plt.title('acc')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()

#loss_acc:  [3.7235297805786134, 0.09529999643564224]