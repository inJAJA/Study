import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Dropout
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


def split_xy2(dataset, time_steps, y_column):
    x_data, y_data = list(), list()
    for j in np.arange(0, 1050000, 375):
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

x_data = x_data.reshape(x_data.shape[0]*x_data.shape[1], 5*4)
y_data = y_data.reshape(y_data.shape[0]*y_data.shape[1], 4)

#2. model_lstm
model = RandomForestRegressor()
model.fit(x_data, y_data)

x1 = model.predict(x_data)

for i in np.arange(369, 370*2800, 370):
    xx = []
    xx.append(x1[i,:])

print(xx.shape)
'''
x_train, x_test, y_train, y_test = train_test_split(x1, y, random_state = 33, )


#3. model_dense
model = Sequential()
model.add(Dense(10, input_shape =(4, )))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation = 'relu'))

model.compile(loss= 'mse', optimizer = 'adam', metrics=['mse'])
model.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_split=0.2)
'''
