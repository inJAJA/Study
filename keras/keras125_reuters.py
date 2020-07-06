from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt       

#1. data
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = 1000, test_split = 0.2) # 제일 많이 쓴는 단어 1000개 가져오기

print(x_train.shape, x_test.shape)  # (8982,) (2246,)
print(y_train.shape, y_test.shape)  # (8982,) (2246,)


print(x_train[0])
print(y_train[0])


print(len(x_train[0]))   # 87


# y의 카테고리 개수 출력
category = np.max(y_train) + 1  # index가 0부터 시작함으로 + 1 해줌
print('카테고리 :', category)    # 카테고리 : 46 (0 ~ 45)

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)


y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print(bbb)
print(bbb.shape)                 # (46,)

# 주간 과제: groupby()의 사용법 숙지할 것

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen = 100, padding = 'pre')
x_test = pad_sequences(x_test, maxlen = 100, padding = 'pre')


# print(len(x_train[0]))            # 100
# print(len(x_train[-1]))           # 100

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)   # (8982, 100) (2246, 100)


#2. model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten

model = Sequential()
# model.add(Embedding(1000, 128, input_length = 100))
model.add(Embedding(1000, 128))

model.add(LSTM(64))
model.add(Dense(46, activation = 'softmax'))

# model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
               metrics = ['acc'])

history = model.fit(x_train, y_train, batch_size= 100, epochs = 10,
                     validation_split = 0.2)

acc = model.evaluate(x_test, y_test)[1]
print('acc :', acc)                               # acc : 0.6313446164131165
                                                  # acc : 0.6763134598731995

# 그림을 그리자
y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker = '.', c = 'red', label = 'TestSet Loss')
plt.plot(y_loss, marker = '.', c = 'blue', label = 'TraintSet Loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
