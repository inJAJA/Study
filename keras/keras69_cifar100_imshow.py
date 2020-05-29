# cifar10 색상이 들어가 있다.
from keras.datasets import cifar100
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

#1. data
(x_train, y_train),(x_test, y_test) = cifar100.load_data()

print(x_train[0])
print('y_train[0] :',y_train[0])

print(x_train.shape)                # (50000, 32, 32, 3)
print(x_test.shape)                 # (10000, 32, 32, 3)
print(y_train.shape)                # (50000, 1)
print(y_test.shape)                 # (10000, 1)


plt.imshow(x_train[3])
plt.show()


from keras.callbacks import EarlyStopping, ModelCheckpoint
# EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience = 50)
# Checkpoint
modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'
checkpoint = ModelChckpoint(filepath = modelpath, monitor = 'val_loss',
                            save_best_only = True)


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])