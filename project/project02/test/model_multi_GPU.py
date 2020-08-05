import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import InceptionV3, MobileNet, Xception
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import HDF5Matrix
import time
import h5py
# import efficientnet.tfkeras as efn 
import tensorflow as tf
# tf.compat.v1.keras.applications.EfficientNetB0

start = time.time()

# load data
x = np.load('D:/data/face_image_total.npy')
y = np.load('D:/data/face_label_total.npy')


# print(x.shape) # (7160, 112, 112, 3)
# print(y.shape) # (7160, 11)
print('data_load 걸린 시간 :', time.time() - start)
print('======== data load ========')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state = 66)

# model
def cnn_model():
    takemodel = Xception(include_top = False, input_shape = (1024, 1024, 3))
    takemodel.trainable = False
    model = Sequential()
    model.add(takemodel)
    model.add(Flatten())
    model.add(Dense(12, activation = 'softmax'))

    model.summary()

    return model

#----------------------------------------------------------------------------
# Multi-GPU model
from tensorflow.keras.utils import multi_gpu_model
model = cnn_model()
model = multi_gpu_model(model, gpus = 2)                    # gpu 2개 사용
model.summary()
#----------------------------------------------------------------------------

cp = ModelCheckpoint('D:/checkpoint/best_32.hdf5', monitor = 'val_loss',
                        save_best_only = True, save_weights_only = False)
es = EarlyStopping(monitor= 'val_loss', patience = 25, verbose =1)

#3. compile, fit
model.compile(optimizer = Adam(1e-4), loss = 'sparse_categorical_crossentropy', metrics = ['acc'])                             
hist = model.fit(x, y, epochs = 100,  verbose = 1,
                    shuffle = True,
                    callbacks = [es, cp])


#4. evaluate
loss_acc = model.evaluate(x, y, batch_size = 32)
print('loss_acc: ' ,loss_acc)

end = time.time()
print('총 걸린 시간 :', end-start)

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
plt.ylabel('acc')
plt.legend()

plt.show()
