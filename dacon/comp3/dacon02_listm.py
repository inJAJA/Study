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
from keras.models import load_model
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

print(x_pred.shape)

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


model = load_model('./dacon/comp3/model_save_lstm.h5') 

def x1_data(data):
    xx = []
    for i in np.arange(369, len(data), 370):
        xx.append(data[i,:,:])
    return np.array(xx)

x1 = x1_data(x_data)
x_pred1 = x1_data(x_pred_data)
print(x1.shape)
print(x_pred1.shape)


x2 = model.predict(x1)
x_pred2 = model.predict(x_pred1)

print(x2.shape)
print(x_pred2.shape)


x_train, x_test, y_train, y_test = train_test_split(x2, y, random_state = 33, )
"""
#3. model_dense
deep = Sequential()
deep.add(Dense(10, input_shape =(4, )))
deep.add(Dropout(0.2))
deep.add(Dense(50, activation = 'relu'))
deep.add(Dropout(0.2))
deep.add(Dense(100, activation = 'relu'))
deep.add(Dropout(0.2))
deep.add(Dense(120, activation = 'relu'))
deep.add(Dropout(0.2))
deep.add(Dense(200, activation = 'relu'))
deep.add(Dropout(0.2))
deep.add(Dense(300, activation = 'relu'))
deep.add(Dropout(0.2))
deep.add(Dense(150, activation = 'relu'))
deep.add(Dropout(0.2))
deep.add(Dense(90, activation = 'relu'))
deep.add(Dropout(0.2))
deep.add(Dense(40, activation = 'relu'))
deep.add(Dropout(0.2))
deep.add(Dense(4, activation = 'relu'))

# es
es = EarlyStopping(monitor = 'val_loss', patience = 50, verbose =1)

deep.compile(loss= 'mse', optimizer = 'adam', metrics=['mse'])
deep.fit(x_train, y_train, epochs = 500, batch_size = 128, validation_split=0.2)

loss_mse = deep.evaluate(x_test, y_test, batch_size = 128)
print('loss_mse: ', loss_mse)

y1_pred = deep.predict(x_test)
"""


params = {                            
    'rf__n_estimators':[100, 200],                             # : 결정트리의 갯수를 지정, defalut = 10
    'rf__max_depth':[5, 10, 20, 40, 50],                                 # : 트리의 최재 깊이, dsfalut = None
    'rf__min_samples_split': [2, 5, 10],                       # : 노드를 분할하기 위한 최소한의 샘플 수, 작을 수록 과적합 가능성 증가
    'rf__min_samples_leaf': [5, 8, 10],                         # : 리프노드가 되기 위해 필요한 최소한의 샘플 개수, 과적합 제어
    'rf__max_features':[2, 'sqrt', 'log2'],  # auto = sqrt     # : 최적의 분할을 위해 고려할 최대 feature개수, default = 'auto'
    # 'rf__max_leaf_nodes' : [],
    # 'rf__min_impurity_decrease':[]
    # 'rf__min_impurity_split': []
}

from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', RobustScaler()), ('rf', RandomForestRegressor())])

model2 = RandomizedSearchCV(pipe, params, cv = 3)

model2.fit(x_train, y_train)

print('best: ', model2.best_params_)

score = model2.score(x_test, y_test)
print('score: ', score)

y_pred = model2.predict(x_pred2)
print(y_pred.shape)
y1_pred = model2.predict(x_test)

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

print(kaeri_metric(y_test, y1_pred))
print(E1(y_test, y1_pred))
print(E2(y_test, y1_pred))

