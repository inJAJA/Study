from keras.datasets import cifar10
from keras.applications import VGG16, VGG19, Xception, ResNet101, ResNet101V2, ResNet152
from keras.applications import ResNet152V2, ResNet50, ResNet50V2, InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobile

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, MaxPool2D, Flatten, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

#2. model
resnet = ResNet50V2(include_top = False, input_shape = (32, 32, 3))

# vgg.summary()

model = Sequential()
model.add(resnet)
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

model.summary()

#3. compile, fit
model.compile(optimizer = Adam(1e-4), loss = 'sparse_categorical_crossentropy', metrics = ['acc'])                             
hist = model.fit(x_train, y_train, epochs = 20, batch_size = 32, verbose = 1, 
                 validation_split =0.3 ,shuffle = True)


#4. evaluate
loss_acc = model.evaluate(x_test, y_test, batch_size = 32)
print('loss_acc: ' ,loss_acc)

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

# loss_acc:  [1.00118753926754, 0.769599974155426]