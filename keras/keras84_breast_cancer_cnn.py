from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()

x = breast_cancer.data
y = breast_cancer.target

print(x.shape)                     # (569, 30)
print(y.shape)                     # (569, )

# x
# scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
# reshape
x = x.reshape(x.shape[0], 5, 3, 2)

# y
# 0과 1로만 이루어졌는지 확인
for i in range(len(y)):
    if (y[i]==0) or (y[i]==1):
        a =+ 0
    else:
        a =+ 1
print(a)                             # 0
# 이진 분류

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 30,
                                                    train_size = 0.8)



#2. model
model = Sequential()
model.add(Conv2D(10,(3, 3), input_shape =(5, 3, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(50,(3, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(Conv2D(80,(3, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(Conv2D(100, (3, 3),activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(Conv2D(150, (3, 3),activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(Conv2D(120, (3, 3),activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(Conv2D(80,(3, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(Conv2D(40,(3, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(Conv2D(20,(3, 3),activation = 'relu', padding = 'same'))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))


# callbacks
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
# earlystopping
es = EarlyStopping(monitor = 'val_loss', patience = 50, verbose = 1 )
# tensorboard
ts_board = TensorBoard(log_dir = 'graph', histogram_freq = 0,
                        write_graph = True, write_images = True)
# modelcheckpotin
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
ckpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                          save_best_only = True)


#3. compile, fit
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32,
                validation_split = 0.2, verbose = 2,
                callbacks = [es])


#4. evaluate, predict
# evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)
print('loss: ', loss )
print('acc: ', acc)

# graph
import matplotlib.pyplot as plt
plt.figure(figsize = (10, 5))

# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], c= 'red', marker = '^', label = 'loss')
plt.plot(hist.history['val_loss'], c= 'cyan', marker = '^', label = 'val_loss')
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], c= 'red', marker = 'o', label = 'acc')
plt.plot(hist.history['val_acc'], c= 'cyan', marker = 'o', label = 'val_acc')
plt.title('accuarcy')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()

plt.show()

"""
loss:  0.08278497759448855
acc:  0.9649122953414917
"""