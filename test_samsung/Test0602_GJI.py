## 데이터
#1. 삼성 : 509
#2. 하이트 : 509 ,( 0행 결측치)

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import re

sm = pd.read_csv('D:/Study/data/csv/samsung.csv', index_col = 0, header= 0, sep = ',',encoding = 'cp949')
h = pd.read_csv('D:/Study/data/csv/hite.csv', index_col =0, header= 0, sep = ',', encoding = 'cp949')

print(sm.head())
print(h.head())

print(sm.shape)                 # (700, 1)
print(h.shape)                 # (720, 5)

print(sm.tail())                # Nan
print(h.tail())                # Nan


sm = sm.dropna(how='all')
h = h.dropna(how='all')
# h.iloc[0, 1:] = h.iloc[0, 1:].fillna(value = str(0))
# print(h.iloc[0, 1:])
h1= h.iloc[1:, :].copy()

for i in range(len(h1.index)):
    for j in range(len(h1.iloc[i])):
        h1.iloc[i,j] = int(h1.iloc[i,j].replace(',','') )
                                                        
print(h1[['고가','저가','종가','거래량']].mean())         # 평균 값  
                                                        # 고가    22747.834646
                                                        # 저가    22061.318898
                                                        # 종가    22415.846457
                                                        # 거래량    338999.137795

h.iloc[0, 1] = '22748'
h.iloc[0, 2] = '22061'
h.iloc[0, 3] = '22416'
h.iloc[0, 4] = '339000'


print(sm.shape)                 # (509, 1)
print(h.shape) 

print('----------')
print(sm.head())
print(h.head())

sm = sm.sort_values('일자',ascending = True)
h = h.sort_values('일자',ascending = True)

print(sm.head())
print(h.head())

print(sm.shape)                # (509, 1)
print(h.shape)                 # (509, 5)


for i in range(len(sm.index)): 
    sm.iloc[i,0] = int(sm.iloc[i,0].replace(',',''))
  

for i in range(len(h.index)):
    for j in range(len(h.iloc[i])):
        h.iloc[i,j] = int(h.iloc[i,j].replace(',','') )


np_sm = sm.values
np_h = h.values

np.save('D:/Study/test/samsung_data.npy', arr = np_sm)
np.save('D:/Study/test/hite_data.npy', arr = np_h)


def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number,:]
        tmp_y = dataset[x_end_number : y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy5(np_sm, 5, 1)
x2, y2 = split_xy5(np_h, 5, 1)
print(x1.shape)                    # (504, 5, 1)
print(x2.shape)                    # (504, 5, 5)
print(y1.shape)                    # (504, 1)
print(y2.shape)                    # (504, 1)

print(x2)
x1 = x1.reshape(x1.shape[0], x1.shape[1])
x2 = x2.reshape(x2.shape[0], x2.shape[1]*x2.shape[2])

# scaler
scaler = RobustScaler()
scaler.fit(x1)
x1 = scaler.transform(x1)
scaler.fit(x2)
x2 = scaler.transform(x2)

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0],5, 5)

x1_1 = x1
x2_1 = x2

# split
x1_train, x1_test, y1_train, y1_test = train_test_split(
                                                     x1, y1, train_size =0.8, random_state =30)
x2_train, x2_test, y2_train, y2_test = train_test_split(
                                                     x2, y2, train_size =0.8, random_state =30)

np.save('D:/Study/test/x1_data5.npy', arr = x1)
np.save('D:/Study/test/x2_data5.npy', arr = x2)
np.save('D:/Study/test/y_data5.npy', arr = y1)

#2. model
# 1
input1 = Input(shape = (5,1 ))
dense1 = LSTM(50, activation = 'elu')(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(150, activation = 'elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(200, activation = 'elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(250, activation = 'elu')(dense1)
dense1 = Dropout(0.2)(dense1)

# 2
input2 = Input(shape = (5, 5))
dense2 = LSTM(100, activation = 'elu')(input2)
dense2 = Dropout(0.2)(dense2)
dense2 = Dense(150, activation = 'elu')(dense2)
dense2 = Dropout(0.2)(dense2)
dense2 = Dense(180, activation = 'elu')(dense2)
dense2 = Dropout(0.2)(dense2)
dense2 = Dense(220, activation = 'elu')(dense2)
dense2 = Dropout(0.2)(dense2)
dense2 = Dense(300, activation = 'elu')(dense2)
dense2 = Dropout(0.2)(dense2)

# merge
merge1 = concatenate([dense1, dense2])
middle1 = Dense(400, activation = 'elu')(merge1)
middle1 = Dropout(0.2)(middle1)
middle1 = Dense(300, activation = 'elu')(middle1)
middle1 = Dropout(0.2)(middle1)
middle1 = Dense(200, activation = 'elu')(middle1)
middle1 = Dropout(0.2)(middle1)
middle1 = Dense(100, activation = 'elu')(middle1)
middle1 = Dropout(0.2)(middle1)

# output
output1 = Dense(1, activation='relu')(middle1)

model = Model(inputs = [input1, input2], outputs = output1)

model.summary

# model_save
model.save('D:/Study/test/test0622_model5.h5')

# callbacks
# Earlystopping
es = EarlyStopping(monitor= 'val_loss', patience= 50, verbose= 1, mode = 'auto')
# Modelcheckpoint
modelpath = 'D:/Study/test/test0622_check5_{epoch:02d}-{val_loss:.4f}.hdf5'
point = ModelCheckpoint(filepath=modelpath, monitor= 'val_loss', 
                        save_best_only= True, save_weights_only= False)

#3. compile, fit
model.compile( loss = 'mse', optimizer = 'adam', metrics = ['mse'])
hist = model.fit([x1_train, x2_train], y1_train, epochs = 500, batch_size = 64,
                 validation_split= 0.2, verbose =2,
                callbacks= [es, point])

model.save_weights('D:/Study/test/test0622_weights5.h5')

#4. evaluate, predict
loss, mse = model.evaluate([x1_test, x2_test], y1_test, batch_size= 64)
print('loss: ', loss)
print('mse:', mse)


y_pred = model.predict([x1_test, x2_test])

from sklearn.metrics import mean_squared_error, r2_score
def rmse(test, predict):
    return np.sqrt(mean_squared_error(y1_test, y_pred))
print('RMSE: ', rmse(y1_test, y_pred))

r2 = r2_score(y1_test, y_pred)
print('R2: ', r2)
plt.plot(hist.history['loss'],c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], c ='cyan', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.show()

print(x1[-1,:])
a = x1_1[-1, :].reshape(1, 5, 1)
b = x2_1[-1, :].reshape(1, 5,5)
y_predict = model.predict([a, b])
print('2020.06.03 :', y_predict)

'''
loss:  1493496.3861386138
mse: 1493496.375
RMSE:  1222.087036206129
R2:  0.9330235314099963
[[0.375     ]
 [0.42358079]
 [0.79912664]
 [0.6069869 ]
 [0.74672489]]
2020.06.03 : [[51797.684]]
'''