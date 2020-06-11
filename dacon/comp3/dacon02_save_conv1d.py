import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from keras.callbacks import EarlyStopping

#1. data
x = pd.read_csv('./data/dacon/comp3/train_features.csv', index_col =0, header = 0)
y = pd.read_csv('./data/dacon/comp3/train_target.csv', index_col = 0, header = 0)
test = pd.read_csv('./data/dacon/comp3/test_features.csv', index_col = 0, header = 0)


x = x.drop('Time', axis =1)
test = test.drop('Time', axis =1)


x = x.values
y = y.values
x_pred = test.values

print(np.arange(0, len(x), 375))
print(np.arange(0, 370))

print(y.shape)

def split_xy2(dataset, time_steps, y_column):
    x_data, y_data = list(), list()
    for j in np.arange(0, len(dataset), 375):
        x, y = list(), list()
        for i in range(0, 370):
            start = i+j
            x_end_number = start + time_steps
            y_end_number = x_end_number + y_column
            tmp_x = dataset[start : x_end_number, :]
            tmp_y = dataset[x_end_number : y_end_number, :]
            x.append(tmp_x)
            y.append(tmp_y)
        x_data.append(x)
        y_data.append(y)   
    return np.array(x_data), np.array(y_data)

x_data, y_data = split_xy2(x, 5, 1)           
print(x_data.shape)                    # (2800, 370, 5, 4)                    
print(y_data.shape)                    # (2800, 370, 1, 4)   

x_pred_data, y_pred_data = split_xy2(x_pred, 5, 1)
print(x_pred_data.shape)               # (700, 370, 5, 4)
print(y_pred_data.shape)               # (700, 370, 1, 4)

x_data = x_data.reshape(x_data.shape[0]*x_data.shape[1], 5, 4)
y_data = y_data.reshape(y_data.shape[0]*y_data.shape[1], 4)

x_pred_data = x_pred_data.reshape(x_pred_data.shape[0]*x_pred_data.shape[1], 5, 4)
y_pred_data = y_pred_data.reshape(y_pred_data.shape[0]*y_pred_data.shape[1], 4)


print(x_data[369,:])
print(y_data[369,:])
print(x_data[370,:])
print(y_data[370, :])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state = 30, train_size = 0.8)

#2. model_lstm
model = Sequential()
model.add(Conv1D(50, 2, input_shape = (5, 4), activation = 'relu'))
model.add(Flatten())
model.add(Dense(100, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation= 'relu'))
# model.add(Dense(100, activation= 'relu'))
# model.add(Dense(100, activation= 'relu'))
model.add(Dense(150, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(4, activation= 'relu'))


es = EarlyStopping(monitor = 'val_loss', patience = 50, verbose = 1)

model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse'])
model.fit(x_train, y_train, epochs = 50, batch_size = 128, validation_split = 0.2,
          callbacks = [es] )

model.save('./dacon/comp3/model_save_conv1d.h5')

loss_mse = model.evaluate(x_test, y_test)
print('loss_mse : ', loss_mse)
