import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Dropout, Input
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.pipeline import Pipeline

from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2)

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

def split_x_pred(dataset, time_steps):
    x = []
    for i in np.arange(0, len(dataset), 375):
        start = i + 370
        x_end_number = start + time_steps
        tmp_x = dataset[start : x_end_number, :]
        x.append(tmp_x)  
    return np.array(x)

x_ml = split_x_pred(x, 5)                 # id_0의 끝에서부터의 5개를 x_predict로 잡기 위한 함수                       
x_ml_pred = split_x_pred(x_pred, 5)       # split_xy2에서 id_0의 끝값이 y값으로 빠짐으로 새로 만듦                 
print(x_ml.shape)
print(x_ml_pred.shape)


x_deep = model.predict(x_ml)
x_deep_pred = model.predict(x_ml_pred)

print(x_deep.shape)
print(x_deep_pred.shape)

x_train, x_test, y_train, y_test = train_test_split( x_deep, y, random_state = 33, train_size = 0.8 )

#2. model
# Deep_learning
def build_model(act = 'relu', drop = 0.2, optimizer = 'adam'):
    inputs = Input(shape = (4, ))
    if act == 'leaky':                                   # leaky가 함수여서 에러가 뜸 
        act = leaky                                      # 이렇게 함수로 바꿔줌
    x = Dense(50, activation = act)(inputs)
    x = Dropout(drop)(x)
    x = Dense(100, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(200, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(100, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation = act)(x)
    output = Dense(4, activation = act)(x)

    model = Model(inputs = inputs, outputs = output)
    model.compile(loss= 'mse', optimizer = optimizer, metrics=['mse'])
    return model
# es
# es = EarlyStopping(monitor = 'val_loss', patience = 50, verbose =1)

def create_hyperparameter():
    batches = [64, 128, 256]
    epochs = [100, 200, 300]
    dropout = np.linspace(0.1, 0.5, 5).tolist()
    activation=  ['relu','leaky', 'elu']
    optimizers = ['rmsprop', 'adam', 'adadelta']
    return {'deep__batch_size': batches, 'deep__epochs':epochs, 'deep__act': activation,
             'deep__drop': dropout,
            'deep__optimizer': optimizers
            }

params = create_hyperparameter()

# wrapper
model = KerasRegressor(build_fn= build_model, verbose =2)

# pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('deep', model)])

# deep.fit(x_train, y_train, epochs = 500, batch_size = 64, validation_split=0.2)

# loss_mse = deep.evaluate(x_test, y_test, batch_size = 64)
# print('loss_mse: ', loss_mse)

# y1_pred = deep.predict(x_test)

"""
# Machine_learning
params = {                            
    'rf__n_estimators':[100, 200],                             # : 결정트리의 갯수를 지정, defalut = 10
    'rf__max_depth':[5, 10, 20, 40, 50],                       # : 트리의 최재 깊이, dsfalut = None
    'rf__min_samples_split': [2, 5, 10],                       # : 노드를 분할하기 위한 최소한의 샘플 수, 작을 수록 과적합 가능성 증가
    'rf__min_samples_leaf': [5, 8, 10],                         # : 리프노드가 되기 위해 필요한 최소한의 샘플 개수, 과적합 제어
    'rf__max_features':[2, 'sqrt', 'log2'],  # auto = sqrt     # : 최적의 분할을 위해 고려할 최대 feature개수, default = 'auto'
    # 'rf__max_leaf_nodes' : [],
    # 'rf__min_impurity_decrease':[]
    # 'rf__min_impurity_split': []
}

from sklearn.pipeline import Pipeline
# pipe = Pipeline([('scaler', RobustScaler()), ('rf', RandomForestRegressor())])

search = RandomizedSearchCV(estimator = pipe, param_distributions= params, cv = 3)
"""
search = RandomizedSearchCV(pipe, params , cv = 3)
search.fit(x_train, y_train)

print('best: ', search.best_params_)

score = search.score(x_test, y_test)
print('score: ', score)

y_pred = search.predict(x_deep_pred)
print(y_pred.shape)
y1_pred = search.predict(x_test)



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

a = np.arange(2800, 3500)
submission = pd.DataFrame(y_pred, a)
submission.to_csv('./dacon/comp3/submission_1.csv', index = True, index_label= ['id'], header = ['X', 'Y', 'M', 'V'])

'''
best:  {'deep__optimizer': 'rmsprop', 'deep__epochs': 150, 'deep__drop': 0.1, 'deep__batch_size': 64, 'deep__act': 'elu'}
score:  -31279.58130580357
(700, 4)
4.07383519838973
6.13556166343452
2.012108733344941
'''