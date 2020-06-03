import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def split_x(seq, size):                    # 
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


#2. 모델 구성
input1 = Input(shape = (5, ))
x1 = Dense(10)(input1)
x1 = Dense(10)(x1)

input2 = Input(shape = (5,))      #  
x2 = Dense(5)(input2)
x2 = Dense(5)(x2)

merge = concatenate([x1, x2])

output = Dense(1)(merge)

model = Model(inputs = [input1, input2], outputs = output)

model.summary()


#3. compile, fit
model.compile(loss ='mse', optimizer = 'adam')
model.fit([x_sam, x_hit], y_sam, epochs = 5)

