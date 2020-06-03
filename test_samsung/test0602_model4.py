import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# Concatenate : class형식
#

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

print(samsung.shape)                                 # (509, 1)
print(hite.shape)                                    # (509, 5)

samsung = samsung.reshape(samsung.shape[0], )        # (509,) :열이 하나인 것은 vector형태가 편하기 때문

samsung = (split_x(samsung, size))
print(samsung.shape)                                 # (504, 6)


x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]
print(x_sam.shape)
print(y_sam.shape)


# standardscaler
scaler = StandardScaler()
scaler.fit(x_sam)
x_sam = scaler.transform(x_sam)
scaler.fit(hite)
x_hit = scaler.transform(hite)



# PCA
pca =PCA(n_components = 1)          # 1열: 각 '행'별로 '열'을 압축 -> 차원 축소
pca.fit(x_hit)                      # hite의 행이 시간별로 이루워져 있기 때문에 PCA를 돌리면 시계열 데이터가 된다(smasung과 같은 형태의 데이터)
x_hit = pca.transform(x_hit).reshape(x_hit.shape[0], )
print(x_hit.shape)                                    # (509, )

# split
x_hit = split_x(x_hit, size)                          # x_sam의 행과 모양 맞춰주기 위함
print(x_hit.shape)                                    # (504, 6)


# LSTM을 위한 reshape
x_sam = x_sam.reshape(x_sam.shape[0], x_sam.shape[1], 1)
x_hit = x_hit.reshape(x_hit.shape[0], x_hit.shape[1], 1)


#2. 모델 구성 : LSTM 1, LSTM 2
input1 = Input(shape = (5, 1))
x1 = LSTM(70)(input1)
x1 = Dropout(0.1)(x1)
x1 = Dense(110)(x1)
x1 = Dropout(0.1)(x1)
x1 = Dense(130)(x1)
x1 = Dropout(0.1)(x1)
x1 = Dense(150)(x1)
x1 = Dropout(0.1)(x1)
x1 = Dense(170)(x1)


input2 = Input(shape = (6, 1))     
x2 = LSTM(50)(input2) 
x2 = Dropout(0.1)(x2)
x2 = Dense(70)(x2)
x2 = Dropout(0.1)(x2)
x2 = Dense(90)(x2)
x2 = Dropout(0.1)(x2)
x2 = Dense(130)(x2)
x2 = Dropout(0.1)(x2)
x2 = Dense(150)(x2)

       
merge = Concatenate()([x1, x2])            # class형태라 ()필요
# merge = concatenate([x1, x2])
middle = Dense(80, name = 'merge')(merge)
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
model.compile(loss ='mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x_sam, x_hit], y_sam, epochs = 100, batch_size = 32)

