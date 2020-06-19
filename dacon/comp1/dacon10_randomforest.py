import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from keras.layers.merge import concatenate  
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import warnings                                
warnings.filterwarnings('ignore')  

#1. data
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col= 0, header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= 0 , header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col= 0 , header = 0)

print('train.shape: ', train.shape)              # (10000, 75)  = x_train, test
print('test.shape: ', test.shape)                # (10000, 71)  = x_predict
print('submission.shape: ', submission.shape)    # (10000, 4)   = y_predict

print(train.isnull().sum())                      
print(train.isnull().sum())   

test.filter(regex='_src$',axis=1).head().T.plot() # .filter : 원하는 값만 골라내기
plt.show()                                        # regex = :정규 표현식
                                                  # .T : 열의 값들로 볼수 있게 함

test.filter(regex='_dst$',axis=1).head().T.plot()
plt.show()


train = train.interpolate(axis = 0)                  
test = test.interpolate(axis =0)

train.replace(0, np.nan)                          # 0의 값들을 채워주기 위해서 NaN값으로 변환
test.replace(0, np.nan)

train = train.fillna(method = 'bfill')
test = test.fillna(method = 'bfill')

train_scr = train.filter(regex='_src$',axis=1)     # 열이 scr인 것들만 걸러냄
train_dst = train.filter(regex='_dst$',axis=1)     # 열이 dst인 것들만 걸러냄
print('scr:', train_scr.shape)                     # scr: (10000, 35)
print('dst:', train_dst.shape)                     # dst: (10000, 35)

test_scr = test.filter(regex='_src$',axis=1)     
test_dst = test.filter(regex='_dst$',axis=1)
print('scr:', test_scr.shape)                      # scr: (10000, 35)
print('dst:', test_dst.shape)                      # dst: (10000, 35)

# print(test_scr)


train_scr = train_scr.values                       # numpy형식으로 변경
train_dst = train_dst.values
test_scr = test_scr.values
test_dst = test_dst.values

train_gap = train_scr - train_dst                  # gap = scr -dst
test_gap = test_scr - test_dst     


# x, y
x = train.iloc[:, : -4].values
y = train.iloc[:, -4:].values
x_pred = test.values

# x, x_pred, y
x = np.hstack((x, train_gap))
x_pred = np.hstack((x_pred, test_gap))
y = train.iloc[:, -4:].values 

x_train, x_test, y_trian, y_test = train_test_split(x, y, random_state = 33, train_size = 0.8)

params ={
    'max_depth': [2, 3, 4], 'n_estimators': [600, 700, 900]
}

model = RandomForestRegressor()

search = RandomizedSearchCV(model, params, cv = 3)

search.fit(x_train, y_trian)

print(search.best_params_)

score = search.score(x_test, y_test)

y_pred1 = search.predict(x_test)
print('mae: ',mean_absolute_error(y_test, y_pred1))

y_pred = search.predict(x_pred)

# submission
a = np.arange(10000,20000)
submission = pd.DataFrame(y_pred, a)
submission.to_csv('D:/Study/dacon/comp1/sub_rf_gap.csv',index = True, header=['hhb','hbo2','ca','na'],index_label='id')