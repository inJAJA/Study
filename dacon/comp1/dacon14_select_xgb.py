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

leaky = LeakyReLU(alpha = 0.2)

#1. data
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col= 0 , header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= 0 , header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col= 0 , header = 0)

print('train.shape: ', train.shape)              # (10000, 75)  = x_train, test
print('test.shape: ', test.shape)                # (10000, 71)  = x_predict
print('submission.shape: ', submission.shape)    # (10000, 4)   = y_predict

# load_gap
train_gap = np.load('./dacon/comp1/data/train_gap.npy')
test_gap = np.load('./dacon/comp1/data/test_gap.npy')
print(train_gap.shape)                                   # (10000, 35)
print(test_gap.shape)                                    # (10000, 35)

# rho
train_rho = train.iloc[:, 0]   
test_rho = test.iloc[:, 0]                        
y = train.iloc[:, -4:]
print(train_rho.shape)                           # (10000, )
print(test_rho.shape)                            # (10000, )
print(y.shape)                                   # (10000, 4)

x_rho = train_rho.values.reshape(-1, 1)
x_pred_rho = test_rho.values.reshape(-1, 1)
y = y.values

# ratio
x_ratio = np.load('./dacon/comp1/data/train_ratio.npy')
x_pred_ratio = np.load('./dacon/comp1/data/test_ratio.npy')

train_scr_fft = np.load('./dacon/comp1/data/train_scr_fft.npy')
test_scr_fft = np.load('./dacon/comp1/data/test_scr_fft.npy')

train_dst_fft = np.load('./dacon/comp1/data/train_dst_fft.npy')
test_dst_fft = np.load('./dacon/comp1/data/test_dst_fft.npy')

# # abs
# train_scr_fft = np.abs(train_scr_fft)
# test_scr_fft = np.abs(test_scr_fft)

# train_dst_fft = np.abs(train_dst_fft)
# test_dst_fft = np.abs(test_dst_fft)

train_fft = np.abs(train_scr_fft - train_dst_fft)
test_fft = np.abs(test_scr_fft - test_dst_fft)

# np.hstack
x = np.hstack((x_rho, x_ratio, train_fft))
x_pred = np.hstack((x_pred_rho, x_pred_ratio, test_fft))

print(x.shape)                                   # (10000, 10)
print(x_pred.shape)                              # (10000, 10)  

# # scaler
# scaler = StandardScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# x_pred = scaler.transform(x_pred)

# # pca
# pca = PCA(n_components= 36)
# pca.fit(x)
# x = pca.transform(x)
# x_pred = pca.transform(x_pred)


# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size =0.8,
                                                    shuffle = True, random_state = 66)

#2. feature_importance
# xgb = XGBRegressor(gpu_id = 0, tree_method = 'gpu_hist')
xgb = XGBRegressor(n_jobs = -1)

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
        
        select_x_train = selection.transform(x_train)
        select_x_test = selection.transform(x_test)
        select_x_pred = selection.transform(x_pred)
        print(select_x_train.shape[1])
        
        parameter = {
            'n_estimators': [100, 200, 400],
            'learning_rate' : [0.05, 0.07, 0.1],
            'colsample_bytree': [ 0.7, 0.8, 0.9],
            'colsample_bylevel':[ 0.7, 0.8, 0.9],
            'max_depth': [4, 5, 6]
        }
    
        # search = RandomizedSearchCV( XGBRegressor(gpu_id = 0, tree_method = 'gpu_hist'), parameter, cv =5)
        search = RandomizedSearchCV( XGBRegressor(), parameter, cv =5)
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
        submission.to_csv('./dacon/comp1/sub/select_XGB03_%i_%.5f.csv'%(i, mae),index = True, header=['hhb','hbo2','ca','na'],index_label='id')
