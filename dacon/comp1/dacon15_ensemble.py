import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.multioutput import MultiOutputRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from keras.layers.merge import concatenate
from keras.wrappers.scikit_learn import KerasRegressor  

from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2)

#1. data
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col= 0 , header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= 0 , header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col= 0 , header = 0)

# rho2
train_rho2 = np.load("./dacon/comp1/data/train_rho2.npy") 
test_rho2 = np.load("./dacon/comp1/data/test_rho2.npy")                       

train_rho2 = train_rho2.reshape(-1, 1)
test_rho2 = test_rho2.reshape(-1, 1)

# ratio
train_ratio = np.load('./dacon/comp1/data/train_ratio.npy')
test_ratio = np.load('./dacon/comp1/data/test_ratio.npy')

#fft
train_scr_fft = np.load('./dacon/comp1/data/train_scr_fft.npy')
test_scr_fft = np.load('./dacon/comp1/data/test_scr_fft.npy')

train_dst_fft = np.load('./dacon/comp1/data/train_dst_fft.npy')
test_dst_fft = np.load('./dacon/comp1/data/test_dst_fft.npy')

# imag
train_scr_fft_imag = np.load('./dacon/comp1/data/train_scr_fft_imag.npy')
test_scr_fft_imag = np.load('./dacon/comp1/data/test_scr_fft_imag.npy')

train_dst_fft_imag = np.load('./dacon/comp1/data/train_dst_fft_imag.npy')
test_dst_fft_imag = np.load('./dacon/comp1/data/test_dst_fft_imag.npy')


# np.hstack
x1 = np.hstack((train_rho2, train_ratio))
x1_pred = np.hstack((test_rho2, test_ratio))

x2 = np.hstack(( train_scr_fft, train_dst_fft)).reshape(10000, -1, 1)
x2_pred = np.hstack(( test_scr_fft, test_dst_fft)).reshape(10000, -1, 1)

x3 = np.hstack(( train_scr_fft_imag, train_dst_fft_imag)).reshape(10000, -1, 1)
x3_pred = np.hstack(( test_scr_fft_imag, test_dst_fft_imag)).reshape(10000, -1, 1)

print(x1.shape)
print(x2.shape)
print(x3.shape)

# y_data
y = train.iloc[:, -4:]
y = y.values


# scaler
scaler1 = MinMaxScaler()
scaler1.fit(x1)
train_rho2 = scaler1.transform(x1)
test_rho2 = scaler1.transform(x1_pred)

scaler2 = RobustScaler()
scaler2.fit(x2)
train_ratio = scaler2.fit(x2)
test_ratio = scaler2.fit(x2_pred)


scaler2.fit(x3)
train_fft = scaler2.transform(x3)
test_fft = scaler2.transform(x3_pred)

# train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1, x2, x3, y, random_state = 66, train_size = 0.8
)


act = 'relu'

#2. model
input1 = Input(shape = (36, ))
x1 = Dense(80, activation = act)(input1)
x1 = Dropout(0.2)(x1)
x1 = Dense(120, activation = act)(x1)
x1 = Dropout(0.2)(x1)
x1 = Dense(200, activation = act)(x1)
x1 = Dropout(0.2)(x1)

input2 = Input(shape = (34, 1))
x2 = LSTM(80, activation = act)(input2)
x2 = Dropout(0.2)(x2)
x2 = Dense(120, activation = act)(x2)
x2 = Dropout(0.2)(x2)
x2 = Dense(200, activation = act)(x2)
x2 = Dropout(0.2)(x2)

input3 = Input(shape = (34, 1))
x3 = LSTM(80, activation = act)(input3)
x3= Dropout(0.2)(x3)
x3 = Dense(120, activation = act)(x3)
x3= Dropout(0.2)(x3)
x3 = Dense(200, activation = act)(x3)
x3= Dropout(0.2)(x3)


merge = concatenate([x1, x2, x3])
middle = Dense(400, activation = act)(merge)
middle = Dropout(0.2)(middle)
middle = Dense(300, activation = act)(middle)
middle = Dropout(0.2)(middle)
middle = Dense(150, activation = act)(middle)
middle = Dropout(0.2)(middle)

outputs = Dense(4, activation = act)(middle)
model = Model(inputs = [input1, input2, input3], outputs = outputs)

# earlystopping
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 50) 

# modelcheckpoint
modelpath = './dacon/comp1/save/dacon15_{epoch:02d} - {val_loss:.4f}.hdf5'                          # 파일 경로, 파일명 설정
                                 
checkpoint = ModelCheckpoint(filepath= modelpath, monitor='val_loss',
                            save_best_only = True, mode = 'auto')

#3. compile, fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit([x1_train, x2_train, x3_train], y_train, epochs = 500, batch_size = 64, verbose = 2,
         validation_split = 0.2, callbacks = [es, checkpoint])

#4. evaluate, predict
loss_mae = model.evaluate([x1_test, x2_test, x3_test],y_test, batch_size = 128)
print('loss_mae: ', loss_mae)

y_pred = model.predict([x1_pred, x2_pred, x3_pred])
print('y_pred: ', y_pred)

# submission
a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./dacon/comp1/sub/dacon15.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')