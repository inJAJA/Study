# 시계열에서 시작 시간이 맞지 않을 경우 '0'으로 채운다.
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2)

import warnings                                
 
warnings.filterwarnings('ignore')  

#1. data
features = pd.read_csv('./data/dacon/comp3/train_features.csv', index_col =0, header = 0)
target = pd.read_csv('./data/dacon/comp3/train_target.csv', index_col = 0, header = 0)
test = pd.read_csv('./data/dacon/comp3/test_features.csv', index_col = 0, header = 0)

dataset = [features, test]

def isnull_data(data):
    none = data.replace(0, np.nan, inplace = False)
    s1, s2, s3, s4 = [],[],[],[]
    for i in np.arange(0, len(data), 375):
        end = i + 375
        sum1 = none.iloc[i: end, 1].isnull().sum()
        sum2 = none.iloc[i: end, 2].isnull().sum()
        sum3 = none.iloc[i: end, 3].isnull().sum()
        sum4 = none.iloc[i: end, 4].isnull().sum()

        s1.append(sum1)
        s2.append(sum2)
        s3.append(sum3)
        s4.append(sum4)

    return np.array([s1, s2, s3, s4])

train_time = isnull_data(features).reshape(-1, 4)
test_time = isnull_data(test).reshape(-1, 4)

print(train_time[-1, ])

print(train_time.shape)    # (2800, 4)
print(test_time.shape)     # (700, 4)

y_position = target.loc[:, ['X','Y']]

print(y_position.shape)    # (2800, 2)

# data_save
np.save('./dacon/comp3/trian_time.npy', arr= train_time)
np.save('./dacon/comp3/test_time.npy', arr= test_time)


# data_time
x = train_time
x_pred = test_time
y = y_position.values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 33, train_size = 0.8)

# data_rest
x_all = features.values.reshape(-1, 375*5)
x_all_pred = test.values.reshape(-1, 375*5)
y_rest = target.loc[:,['M','V']]

x2_train, x2_test, y2_train, y2_test = train_test_split(x_all, y_rest, random_state = 12, 
                                                        train_size = 0.8)


kfold = KFold(n_splits= 3, shuffle = True)

model = RandomForestRegressor()

params = {
    'max_depth':[3, 4, 5],
    'n_estimators':[100, 200, 300]
}

search = RandomizedSearchCV(model, params, cv = kfold)
search2 = RandomizedSearchCV(model, params, cv = kfold)

multi = MultiOutputRegressor(search)
multi2 = MultiOutputRegressor(search2)

# fit
multi.fit(x_train, y_train)
multi2.fit(x2_train, y2_train)

# print(search.best_estimator_)

# evaluate
y1_pred = multi.predict(x_test)
y2_pred = multi2.predict(x2_test)

print('R2: ', r2_score(y_test, y1_pred))
print('R2: ', r2_score(y2_test, y2_pred))


# predict
y_pred1 = multi.predict(x_pred)
y_pred2 = multi2.predict(x_all_pred)

y_pred = pd.DataFrame({
  'id' : np.arange(2800, 3500),
  'X': y_pred1[:,0],
  'Y': y_pred1[:, 1],
  'M': y_pred2[:, 0],
  'V':y_pred2[:, 1]
})
y_pred.to_csv('./dacon/comp3/sub_time_xgb.csv', index = False )


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

print(kaeri_metric(y_test, y1_pred))
print(E1(y_test, y1_pred))
print(E2(y_test, y1_pred))

print(kaeri_metric(y_test, y2_pred))
print(E1(y_test, y2_pred))
print(E2(y_test, y2_pred))