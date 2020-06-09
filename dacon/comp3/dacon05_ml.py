# 시계열에서 시작 시간이 맞지 않을 경우 '0'으로 채운다.
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline

from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2)

import warnings                                

warnings.filterwarnings('ignore')  

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


print(x.shape)                      # (1050000, 5)
print(x_pred.shape)                 # (700, 375, 4)
print(y.shape)                      # (2800, 4)


x = x.reshape(-1, 375*5)
y = y.reshape(-1, 4)
x_pred = x_pred.reshape(-1, 375*5)



# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 10, train_size = 0.2)


#2. model
param_grid = {
    "regressor__n_estimators" : [100, 200],      
    "regressor__max_depth": [6, 8, 10, 20],      
    "regressor__min_samples_leaf":[3, 5, 7, 10], 
    "regressor__min_samples_split": [2,3, 5],     
    'regressor__max_features': ['auto', 'sqrt', 'log2'] ,                
    "regressor__n_jobs" : [-1]
}     

pipe = Pipeline([("scaler", MinMaxScaler()), ('regressor', RandomForestRegressor())])


search = RandomizedSearchCV(estimator= pipe, param_distributions = param_grid , cv = 5)

search.fit(x_train, y_train)

print(search.best_params_)  

score = search.score(x_test, y_test)
print('score: ', score)


y_pred = search.predict(x_pred)
y_pred1 = search.predict(x_test)


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
submission.to_csv('./dacon/comp3/submission_5.csv', index = True, index_label= ['id'], header = ['X', 'Y', 'M', 'V'])
'''

'''