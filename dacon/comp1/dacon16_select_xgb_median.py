import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from xgboost import XGBRegressor, plot_importance
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.interpolate import interp1d
from keras.layers import LeakyReLU
import pickle 

leaky = LeakyReLU(alpha = 0.2)

#1. data
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col= 0 , header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= 0 , header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col= 0 , header = 0)

print('train.shape: ', train.shape)              # (10000, 75)  = x_train, test
print('test.shape: ', test.shape)                # (10000, 71)  = x_predict
print('submission.shape: ', submission.shape)    # (10000, 4)   = y_predict

# rho2
train_rho2 = np.load("./dacon/comp1/data/train_rho2.npy") 
test_rho2 = np.load("./dacon/comp1/data/test_rho2.npy")                       

x_rho2 = train_rho2.reshape(-1, 1)
x_pred_rho2 = test_rho2.reshape(-1, 1)

# ratio
x_ratio = np.load('./dacon/comp1/data/train_ratio.npy')
x_pred_ratio = np.load('./dacon/comp1/data/test_ratio.npy')

# fourier
train_scr_fft = np.load('./dacon/comp1/data/train_scr_fft2.npy')
test_scr_fft = np.load('./dacon/comp1/data/test_scr_fft2.npy')

train_dst_fft = np.load('./dacon/comp1/data/train_dst_fft2.npy')
test_dst_fft = np.load('./dacon/comp1/data/test_dst_fft2.npy')

print(train_scr_fft.shape)   # (10000, 17)
print(train_dst_fft.shape)   # (10000, 17)

train_fft = train_scr_fft - train_dst_fft
test_fft = test_scr_fft - test_dst_fft

# np.hstack
x = np.hstack((x_rho2, x_ratio, train_fft))
x_pred = np.hstack((x_pred_rho2, x_pred_ratio, train_fft ))

print(x.shape)                                   # (10000, 10)
print(x_pred.shape)                              # (10000, 10)  

# y
y = train.iloc[:, -4:]
y = y.values

# scaler
scaler = RobustScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)

# # pca
# pca = PCA(n_components= 1)
# pca.fit(x)
# x = pca.transform(x)
# x_pred = pca.transform(x_pred)


# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size =0.8,
                                                    shuffle = True, random_state = 66)

#2. feature_importance
# xgb = XGBRegressor(gpu_id = 0, tree_method = 'gpu_hist')

multi_XGB = MultiOutputRegressor(XGBRegressor())
multi_XGB.fit(x_train, y_train)

print(len(multi_XGB.estimators_))   # 4


# print(multi_XGB.estimators_[0].feature_importances_)
# print(multi_XGB.estimators_[1].feature_importances_)
# print(multi_XGB.estimators_[2].feature_importances_)
# print(multi_XGB.estimators_[3].feature_importances_)

    
for i in range(len(multi_XGB.estimators_)):
    selection = SelectFromModel(multi_XGB.estimators_[i], threshold = 'median', prefit = True)
        
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    select_x_pred = selection.transform(x_pred)
        
    parameter = {
        'n_estimators': [200, 300, 400],
        'learning_rate' : [0.05, 0.07, 0.1],
        'colsample_bytree': [ 0.65, 0.75, 0.85],
        'colsample_bylevel':[ 0.75, 0.85, 0.65],
        'max_depth': [4, 5, 6]
    }
    
    # search = RandomizedSearchCV( XGBRegressor(gpu_id = 0, tree_method = 'gpu_hist'), parameter, cv =5)
    xgb = XGBRegressor()
    search = RandomizedSearchCV( xgb, parameter, cv =5)
    multi_search = MultiOutputRegressor(search,n_jobs = -1)

    multi_search.fit(select_x_train, y_train )
        
    y_pred = multi_search.predict(select_x_test)
    mae = mean_absolute_error(y_test, y_pred)
    score =r2_score(y_test, y_pred)
    print("Thresh=%.3f, n = %d, R2 : %.2f%%, MAE : %.3f"%(thres, select_x_train.shape[1], score*100.0, mae))
        
    y_predict = multi_search.predict(select_x_pred)

    # submission
    a = np.arange(10000,20000)
    submission = pd.DataFrame(y_predict, a)
    submission.to_csv('./dacon/comp1/sub/dacon16_%.5f.csv'%(mae),index = True, header=['hhb','hbo2','ca','na'],index_label='id')
'''
'''
