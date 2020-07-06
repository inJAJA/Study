from keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt       

#1. data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 2000) # train:test = 50:50

print(x_train.shape, x_test.shape)  # (25000,) (25000,)
print(y_train.shape, y_test.shape)  # (25000,) (25000,)


# print(x_train[0])
# print(y_train[0])        # 1


print(len(x_train[0]))   # 218 

# x_train내용 보기
word_to_index = imdb.get_word_index() # {'단어': index, }
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key        # {'index' : 단어}
print(' '.join([index_to_word[index] for index in x_train[0]]))
                            

# y의 카테고리 개수 출력
category = np.max(y_train) + 1     # index가 0부터 시작함으로 + 1 해줌
print('카테고리 :', category)       # 카테고리 : 2

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
# print(y_bunpo)                   # [0 1]


y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
# print(bbb)                       # 0    12500   : 부정
                                   # 1    12500   : 긍정
# print(bbb.shape)                 # (2,)



from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen = 111, padding = 'pre')
x_test = pad_sequences(x_test, maxlen = 111, padding = 'pre')


# print(len(x_train[0]))            # 111
# print(len(x_train[-1]))           # 111

# y_train = to_categorical(y_train)  # 이진 분류임으로 원핫인코딩 필요없음
# y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)   # (25000, 111) (25000, 111)


#2. model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, MaxPooling1D

model = Sequential()
# model.add(Embedding(1000, 128, input_length = 111))
model.add(Embedding(2000, 128))

model.add(Conv1D(32, 5, padding = 'valid', activation = 'relu', strides = 1))
model.add(MaxPooling1D(pool_size=4))

model.add(LSTM(32, return_sequences= True))
model.add(LSTM(32))
model.add(Dense(10))
model.add(Dense(1, activation = 'sigmoid'))

# model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
               metrics = ['acc'])

history = model.fit(x_train, y_train, batch_size= 128, epochs = 10,
                     validation_split = 0.1)

acc = model.evaluate(x_test, y_test)[1]
print('acc :', acc)                          # acc : 0.8343600034713745             

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


# 1. imdb 검색해서 데이터 내용 확인
# 2. word_size 전체 데이터 부분 변경해서 최상값 확인
# 3. 주간과제 : groupby() 사용법 숙지할 것
# 4. 인덱스를 단어로 바꿔주는 함수 찾을 것 : .index_to_word[]
