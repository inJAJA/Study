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
import warnings                                

warnings.filterwarnings('ignore') 
leaky = LeakyReLU(alpha = 0.2)

#1. data
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col= 0 , header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= 0 , header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col= 0 , header = 0)

print('train.shape: ', train.shape)              # (10000, 75)  = x_train, test
print('test.shape: ', test.shape)                # (10000, 71)  = x_predict
print('submission.shape: ', submission.shape)    # (10000, 4)   = y_predict

# rho
train_rho = train.iloc[:, 0].values.reshape(-1, 1)
test_rho = test.iloc[:, 0].values.reshape(-1, 1)

# rho2
train_rho2 = np.load("./dacon/comp1/data/train_rho2.npy") 
test_rho2 = np.load("./dacon/comp1/data/test_rho2.npy")                       

train_rho2 = train_rho2.reshape(-1, 1)
test_rho2 = test_rho2.reshape(-1, 1)

# ratio
train_ratio = np.load('./dacon/comp1/data/train_ratio.npy')
test_ratio = np.load('./dacon/comp1/data/test_ratio.npy')

# ratio_0
train_ratio_0 = np.load('./dacon/comp1/data/train_ratio_0.npy')
test_ratio_0 = np.load('./dacon/comp1/data/test_ratio_0.npy')

#fft
train_scr_fft = np.load('./dacon/comp1/data/train_scr_fft.npy')
test_scr_fft = np.load('./dacon/comp1/data/test_scr_fft.npy')

train_dst_fft = np.load('./dacon/comp1/data/train_dst_fft.npy')
test_dst_fft = np.load('./dacon/comp1/data/test_dst_fft.npy')

# imag
train_scr_fft_imag = np.load('./dacon/comp1/data/train_scr_fft_imag.npy')
test_scr_fft_imag = np.load('./dacon/comp1/data/test_scr_fft_imag.npy')

train_dst_fft_imag = np.load('./dacon/comp1/data/train_dst_fft_imag.npy')
test_dst_fft_imag = np.load('./dacon/comp1/data/test_dst_fft_imag.npy')


# RobustScaler
scaler = RobustScaler()
# fft
scaler.fit(train_scr_fft)
train_scr_fft = scaler.transform(train_scr_fft)
test_scr_fft = scaler.transform(test_scr_fft)

scaler.fit(train_dst_fft)
train_dst_fft = scaler.transform(train_dst_fft)
test_dst_fft = scaler.transform(test_dst_fft)

# fft_imag
scaler.fit(train_scr_fft_imag)
train_scr_fft = scaler.transform(train_scr_fft_imag)
test_scr_fft = scaler.transform(test_scr_fft_imag)

scaler.fit(train_dst_fft_imag)
train_dst_fft = scaler.transform(train_dst_fft_imag)
test_dst_fft = scaler.transform(test_dst_fft_imag)

# StandardScaler
stand = StandardScaler()
# ratio
stand.fit(train_ratio)
train_ratio = stand.transform(train_ratio)
test_ratio = stand.transform(test_ratio)

# ratio_0
stand.fit(train_ratio_0)
train_ratio = stand.transform(train_ratio_0)
test_ratio = stand.transform(test_ratio_0)

# np.hstack
x = np.hstack((train_rho2, train_ratio,  train_ratio_0, (train_scr_fft - train_dst_fft), 
               train_scr_fft, train_dst_fft, train_scr_fft_imag, train_dst_fft_imag ))
x_pred = np.hstack((test_rho2, test_ratio, test_ratio_0,  (test_scr_fft - test_dst_fft), 
                    test_scr_fft, test_dst_fft, test_scr_fft_imag, test_dst_fft_imag))

print(x.shape)                                   # (10000, 10)
print(x_pred.shape)                              # (10000, 10)  

# y
y = train.iloc[:, -4:]
y = y.values


# # pca
# pca = PCA(n_components= 1)
# pca.fit(x)
# x = pca.transform(x)
# x_pred = pca.transform(x_pred)


# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size =0.8,
                                                    shuffle = True, random_state = 66)

#2. feature_importance
multi_XGB = MultiOutputRegressor(XGBRegressor(n_jobs = 5))
multi_XGB.fit(x_train, y_train)

print(len(multi_XGB.estimators_))   # 4


# print(multi_XGB.estimators_[0].feature_importances_)
# print(multi_XGB.estimators_[1].feature_importances_)
# print(multi_XGB.estimators_[2].feature_importances_)
# print(multi_XGB.estimators_[3].feature_importances_)


best_mae = 1.3
best_score = 0.45

y_predict = []
for i in range(len(multi_XGB.estimators_)):
    threshold = np.sort(multi_XGB.estimators_[i].feature_importances_)[60:]

    for thres in threshold:
        selection = SelectFromModel(multi_XGB.estimators_[i], threshold = thres, prefit = True)
        
        select_x_train = selection.transform(x_train)
        select_x_test = selection.transform(x_test)
        select_x_pred = selection.transform(x_pred)
        
        parameter = {
            'n_estimators': [450, 550],
            'learning_rate' : [0.065, 0.05],
            'colsample_bytree': [ 0.65, 0.7],
            'colsample_bylevel':[ 0.75, 0.85, 0.65],
            'reg_alpha' : [1],
            'max_depth': [7, 8],
            'scale_pos_weight': [1],
            'reg_lambda' : [1.1]
        }
    
        # search = RandomizedSearchCV( XGBRegressor(gpu_id = 0, tree_method = 'gpu_hist'), parameter, cv =5)
        xgb = XGBRegressor()
        search = RandomizedSearchCV( xgb, parameter, cv = 3, n_jobs = 5)

        search.fit(select_x_train, y_train[:, i], verbose = False, eval_metric = ['rmse', 'mae'],
                                                   eval_set = [(select_x_test, y_test[:, i])])
        
        y_pred = search.predict(select_x_test)
        mae = mean_absolute_error(y_test[:, i], y_pred)
        score =r2_score(y_test[:, i], y_pred)
        print("Thresh=%.3f, n = %d, R2 : %.2f%%, MAE : %.3f"%(thres, select_x_train.shape[1], score*100.0, mae))
        
        y_predict = search.predict(select_x_pred)
        
        if mae < best_mae or score > best_score:
            best_mae = mae
            best_score = score
        # submission
            if i == 0: 
                a = np.arange(10000,20000)
                submission = pd.DataFrame(y_predict, a)
                submission.to_csv('./dacon/comp1/sub/dacon21_col1_%.5f_%.2f.csv'%( best_mae, score*100.0 ),index = True, header=['hhb'],index_label='id')
            elif i == 1:
                a = np.arange(10000,20000)
                submission = pd.DataFrame(y_predict, a)
                submission.to_csv('./dacon/comp1/sub/dacon21_col2_%.5f_%.2f.csv'%( best_mae, score*100.0 ),index = True, header=['hbo2'],index_label='id')
            elif i == 2:
                a = np.arange(10000,20000)
                submission = pd.DataFrame(y_predict, a)
                submission.to_csv('./dacon/comp1/sub/dacon21_col3_%.5f_%.2f.csv'%( best_mae, score*100.0 ),index = True, header=['ca'],index_label='id')
            else:
                a = np.arange(10000,20000)
                submission = pd.DataFrame(y_predict, a)
                submission.to_csv('./dacon/comp1/sub/dacon21_col4_%.5f_%.2f.csv'%( best_mae, score*100.0 ),index = True, header=['na'],index_label='id')
           
            
'''            
# submission
a = np.arange(10000,20000)
submission = pd.DataFrame(y_predict, a)
submission.to_csv('./dacon/comp1/sub/dacon20_3_%.5f_%.2f.csv'%(mae, score*100.0),index = True, header=['hhb','hbo2','ca','na'],index_label='id')
'''