import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def split_x(seq, size):                   
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i: (i+size)]
        aaa.append([j for j in subset])          
    return np.array(aaa)

size = 6                      

#1. 데이터
# npy 불러오기
samsung = np.load('D:/Study/data/samsung.npy', allow_pickle = 'True')
hite = np.load('D:/Study/data/hite.npy', allow_pickle = 'True')

print(samsung.shape)               # (509, 1)
print(hite.shape)                  # (509, 5)

samsung = samsung.reshape(samsung.shape[0], )     # (509,) :열이 하나인 것은 vector형태가 편하다 

samsung = (split_x(samsung, size))
print(samsung.shape)               # (504, 6)

# hite는 y값을 만들 필요가 없다.
# samsung에서만 y값을 생성

x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]

print(x_sam.shape)                 # (504, 5) : 앙상블할 때 행까지 맞춰줘야 함
print(y_sam.shape)                 # (504, )

x_hit = hite[5: 510, :]
print(x_hit.shape)                 # (504, 5)

x_sam = x_sam.reshape(x_sam.shape[0], x_sam.shape[1], 1)

print('x_sam:',x_sam)              # '2020-06-02'의 값이 y값으로 들어갔다.

print(x_sam[-1,:])
print(x_hit[-1, :])

x1_pred = samsung[-5:, 0]
x2_pred = hite[-1,:]
print('samsung_data:', samsung)
print('samsung: ',x1_pred)
print('hite: ',x2_pred)

'''
#2. 모델 구성: LSTM 1, Dense2
input1 = Input(shape = (5, 1))
x1 = LSTM(70)(input1)             # 시계열은 LSTM이 좋다
x1 = Dropout(0.1)(x1)             # 서로가 영향이 있기 때문에
x1 = Dense(110)(x1)
x1 = Dropout(0.1)(x1)
x1 = Dense(130)(x1)
x1 = Dropout(0.1)(x1)
x1 = Dense(150)(x1)
x1 = Dropout(0.1)(x1)
x1 = Dense(170)(x1)


input2 = Input(shape = (5,))      
x2 = Dense(50)(input2)             # 시계열이 아니기 때문에 Dense가 좋다
x2 = Dropout(0.1)(x2)              # 각 열이 하나하나 종류이기 때문에
x2 = Dense(70)(x2)
x2 = Dropout(0.1)(x2)
x2 = Dense(90)(x2)
x2 = Dropout(0.1)(x2)
x2 = Dense(130)(x2)
x2 = Dropout(0.1)(x2)
x2 = Dense(150)(x2)


merge
merge = concatenate([x1, x2])
middle = Dense(80)(merge)
middle = Dropout(0.1)(middle)
middle = Dense(80)(middle)
middle = Dropout(0.1)(middle)
middle = Dense(50)(middle)
middle = Dropout(0.1)(middle)
middle = Dense(30)(middle)
middle = Dropout(0.1)(middle)

output = Dense(1)(middle)

model = Model(inputs = [input1, input2], outputs = output)

model.summary()


#3. compile, fit
model.compile(loss ='mse', optimizer = 'adam')
model.fit([x_sam, x_hit], y_sam, epochs = 5)

'''