#### 과적합 막는 법 #####
# 훈련데이터 늘린다.
# feature를 늘린다
# 정규화 시킨다.

# cifar10 색상이 들어가 있다.
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout, Input
import matplotlib.pyplot as plt

#1. data
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

print(x_train[0])
print('y_train[0] :',y_train[0])

print(x_train.shape)                # (50000, 32, 32, 3)
print(x_test.shape)                 # (10000, 32, 32, 3)
print(y_train.shape)                # (50000, 1)
print(y_test.shape)                 # (10000, 1)


# plt.imshow(x_train[3])
# plt.show()

# # y                                 # 'sparse_categorical_crossentropy'사용으로 원핫 인코딩 필요 없음
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape)                # (50000, 100) 

x_train = x_train.reshape(-1, 32, 32, 3).astype('float32')/225
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32')/225


#2. model
from keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (32, 32, 3)))
model.add(Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))

model.add(Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))

model.add(Conv2D(128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))

model.add(Flatten())
model.add(Dense(256,  activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

                                           
model.compile(optimizer = Adam(1e-4), loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
                            # 0.0001        # one_hot_encoding하지 않고 다중 분류하는 loss


#3. fit                              
hist = model.fit(x_train, y_train, epochs = 30, batch_size = 32, verbose = 1, 
                 validation_split =0.3 ,shuffle = True)


#4. evaluate
loss_acc = model.evaluate(x_test, y_test, batch_size = 32)
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
