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

# x_data
# load
# sqrt rho2
train_rho2 = np.load('./dacon/comp1/data/train_rho2.npy')
test_rho2 = np.load('./dacon/comp1/data/test_rho2.npy')
print(train_rho2.shape)                                   # (10000, 35)
print(test_rho2.shape)                                    # (10000, 35)

train_rho2 = train_rho2.reshape(-1, 1)
test_rho2 = test_rho2.reshape(-1, 1)

# ratio
train_ratio = np.load('./dacon/comp1/data/train_ratio.npy')
test_ratio = np.load('./dacon/comp1/data/test_ratio.npy')

# fourier
train_scr_fft = np.load('./dacon/comp1/data/train_scr_fft.npy')
test_scr_fft = np.load('./dacon/comp1/data/test_scr_fft.npy')

train_dst_fft = np.load('./dacon/comp1/data/train_dst_fft.npy')
test_dst_fft = np.load('./dacon/comp1/data/test_dst_fft.npy')

# fourier_abs
train_scr_fft = np.abs(train_scr_fft)
test_scr_fft = np.abs(test_scr_fft)

train_dst_fft = np.abs(train_dst_fft)
test_dst_fft = np.abs(test_dst_fft)

# np.hstack
train_fft = np.hstack((train_scr_fft, train_dst_fft))
test_fft = np.hstack((test_scr_fft, test_dst_fft))

# y_data
y = train.iloc[:, -4:]
y = y.values

print(type(train_ratio))
print(train_fft.shape)

# scaler
scaler1 = MinMaxScaler()
scaler1.fit(train_rho2)
train_rho2 = scaler1.transform(train_rho2)
test_rho2 = scaler1.transform(test_rho2)

# scaler2 = RobustScaler()
# scaler2.fit(train_ratio)
# train_ratio = scaler2.fit(train_ratio)
# test_ratio = scaler2.fit(test_ratio)


# scaler2.fit(train_fft)
# train_fft = scaler2.transform(train_fft)
# test_fft = scaler2.transform(test_fft)

# train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    train_rho2, train_ratio, train_fft, y, random_state = 66, train_size = 0.8
)
print(x1_train.shape)
print(x2_train.shape)
print(x3_train.shape)
print(y_train.shape)


#2. model
input1 = Input(shape = (1, ))
x1 = Dense(80, activation = 'relu')(input1)
x1 = Dropout(0.2)(x1)
x1 = Dense(120, activation = 'relu')(x1)
x1 = Dropout(0.2)(x1)

input2 = Input(shape = (35, ))
x2 = Dense(80, activation = 'relu')(input2)
x2 = Dropout(0.2)(x2)
x2 = Dense(120, activation = 'relu')(x2)
x2 = Dropout(0.2)(x2)

input3 = Input(shape = (70, ))
x3 = Dense(80, activation = 'relu')(input3)
x3= Dropout(0.2)(x3)
x3 = Dense(120, activation = 'relu')(x3)
x3= Dropout(0.2)(x3)


merge = concatenate([x1, x2, x3])
middle = Dense(80, activation = 'relu')(merge)
middle = Dropout(0.2)(middle)
middle = Dense(50, activation = 'relu')(middle)
middle = Dropout(0.2)(middle)


outputs = Dense(4, activation = 'relu')(middle)
model = Model(inputs = [input1, input2, input3], outputs = outputs)

# earlystopping
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 50) 

# modelcheckpoint
modelpath = './dacon/comp1/save/{epoch:02d} - {val_loss:.4f}.hdf5'                          # 파일 경로, 파일명 설정
                                 
checkpoint = ModelCheckpoint(filepath= modelpath, monitor='val_loss',
                            save_best_only = True, mode = 'auto')

#3. compile, fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit([x1_train, x2_train, x3_train], y_train, epochs = 500, batch_size = 128, verbose = 2,
         validation_split = 0.2, callbacks = [es, checkpoint])

#4. evaluate, predict
loss_mae = model.evaluate([x1_test, x2_test, x3_test],y_test, batch_size = 128)
print('loss_mae: ', loss_mae)

y_pred = model.predict([test_rho2, test_ratio, test_fft])
print('y_pred: ', y_pred)

# submission
a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./dacon/comp1/sub/ensemble.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')