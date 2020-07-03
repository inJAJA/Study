#### 과적합 막는 법 #####
# 훈련데이터 늘린다.        
# feature를 늘린다
# 정규화 시킨다.

'''
# regularizer : 정규화  # 2가지 이상 사용하는 것은 권장하지 않음 = 큰 효과 없어서
: 과적합 줄임
                                           ( 규제 값 )
L1 규제 : 가중치의 절대값 합 = regularizer.l1(l = 0.01)
L2 규제 : 가중ㅊ;의 제곱 합   = regularizer.l2(l = 0.01)

       (규제 내용) 
loss =     L1     * reduce_sum(abs(x))     -> L1 * x의 절대값의 합
loss =     L2     * reduce_sum(square(x))  -> L2 * x의 제곱의 합

# L1
: 작은 W(가중치)들을 거의 0으로 수렴시켜 몇개의 중요한 가중치들만 남긴다.
: sparse model(coding)에 적합

# L2
: 전체적으로 W(가중치)값이 작아지도록 한다. / L1처럼 일부 항의 계수를 0으로 만들지 않음
'''

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
from keras.regularizers import l1, l2, l1_l2          # regularizer 불러오기
model = Sequential()                
model.add(Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (32, 32, 3)))
model.add(Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))  

model.add(Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu', kernel_regularizer=l2(0.001)))
model.add(Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))

model.add(Conv2D(128, kernel_size = 3, padding = 'same', activation = 'relu', kernel_regularizer=l2(0.001)))
model.add(Conv2D(128, kernel_size = 3, padding = 'same', activation = 'relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
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

# loss_acc:  [1.7100680181503296, 0.7109000086784363]