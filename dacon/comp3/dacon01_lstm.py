# 시계열에서 시작 시간이 맞지 않을 경우 '0'으로 채운다.
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Dropout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

#1. data
x = pd.read_csv('./data/dacon/comp3/train_features.csv', index_col =0, header = 0)
y = pd.read_csv('./data/dacon/comp3/train_target.csv', index_col = 0, header = 0)
test = pd.read_csv('./data/dacon/comp3/test_features.csv', index_col = 0, header = 0)


# x = x.drop('Time', axis =1)
# test = test.drop('Time', axis =1)


x = x.values
y = y.values
x_pred = test.values

print(x.shape)                      # (1050000, 4)

# scaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)

x = x.reshape(-1, 375, 5)
x_pred = x_pred.reshape(-1, 375, 5)

print(x.shape)                      # (2800, 375, 4)
print(x_pred.shape)                 # (700, 375, 4)
print(y.shape)                      # (2800, 4)



# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 10, train_size = 0.2)


#2. model
input1 = Input(shape=(375, 5))
x = LSTM(100, activation = 'relu')(input1)
x = Dropout(0.2)(x)
x = Dense(150, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(200, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(250, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(300, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(270, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(210, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(130, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(110, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(50, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(30, activation = 'relu')(x)
x = Dropout(0.2)(x)
output = Dense(4, activation = 'relu')(x)

model = Model(inputs = input1, outputs = output)


# EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 50)

#3. compile, fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 64, 
                 validation_split= 0.2, callbacks = [es])

#4. evaluate
loss_mse = model.evaluate(x_test, y_test, batch_size= 64)
print('loss_mse: ', loss_mse)

y_pred1 = model.predict(x_test)
y_pred = model.predict(x_pred)

# 평가지표
def kaeri_metric(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: KAERI metric
    '''
    
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)


### E1과 E2는 아래에 정의됨 ###

def E1(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: distance error normalized with 2e+04
    '''
    
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)


def E2(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: sum of mass and velocity's mean squared percentage error
    '''
    
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    
    
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))

print(kaeri_metric(y_test, y_pred1))
print(E1(y_test, y_pred1))
print(E2(y_test, y_pred1))


a = np.arange(2800, 3500)
submission = pd.DataFrame(y_pred, a)
submission.to_csv('./dacon/comp3/submission_1.csv', index = True, index_label= ['id'], header = ['X', 'Y', 'M', 'V'])