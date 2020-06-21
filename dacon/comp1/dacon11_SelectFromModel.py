import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from xgboost import XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel


#1. data
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col= 0 , header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= 0 , header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col= 0 , header = 0)

print('train.shape: ', train.shape)              # (10000, 75)  = x_train, test
print('test.shape: ', test.shape)                # (10000, 71)  = x_predict
print('submission.shape: ', submission.shape)    # (10000, 4)   = y_predict


trian = train.interpolate(axis = 0)
test= test.interpolate(axis = 0)

train = train.fillna(train.mean())
test = test.fillna(test.mean())


x = train.iloc[:, :71]                           
y = train.iloc[:, -4:]
print(x.shape)                                   # (10000, 71)
print(y.shape)                                   # (10000, 4)

x = x.values
y = y.values
x_pred = test.values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size =0.8,
                                                    shuffle = True, random_state = 66)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


#2. feature_importance
xgb = XGBRegressor()
multi_XGB = MultiOutputRegressor(xgb)
multi_XGB.fit(x_train, y_train)

print(len(multi_XGB.estimators_))   # 4


# print(multi_XGB.estimators_[0].feature_importances_)
# print(multi_XGB.estimators_[1].feature_importances_)
# print(multi_XGB.estimators_[2].feature_importances_)
# print(multi_XGB.estimators_[3].feature_importances_)

for i in range(len(multi_XGB.estimators_)):
    threshold = np.sort(multi_XGB.estimators_[i].feature_importances_)

    for thres in threshold:
        selection = SelectFromModel(multi_XGB.estimators_[i], threshold = thres, prefit = True)
        
        parameter = {
            'n_estimators': [100, 200, 400],
            'learning_rate' : [0.03, 0.05, 0.07, 0.1],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'colsample_bylevel':[0.6, 0.7, 0.8, 0.9],
            'max_depth': [4, 5, 6]
        }
    
        search = RandomizedSearchCV( XGBRegressor(), parameter, cv =5)

        select_x_train = selection.transform(x_train)

        multi_search = MultiOutputRegressor(search,n_jobs = -1)
        multi_search.fit(select_x_train, y_train )
        
        select_x_test = selection.transform(x_test)

        y_pred = multi_search.predict(select_x_test)
        mae = mean_absolute_error(y_test, y_pred)
        score =r2_score(y_test, y_pred)
        print("Thresh=%.3f, n = %d, R2 : %.2f%%, MAE : %.3f"%(thres, select_x_train.shape[1], score*100.0, mae))
 
        select_x_pred = selection.transform(x_pred)
        y_predict = multi_search.predict(select_x_pred)
        # submission
        a = np.arange(10000,20000)
        submission = pd.DataFrame(y_predict, a)
        submission.to_csv('./dacon/comp1/sub_XG%i_%.5f.csv'%(i, mae),index = True, header=['hhb','hbo2','ca','na'],index_label='id')
