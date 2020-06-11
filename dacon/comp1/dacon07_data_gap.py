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
print(train_gap)

np.save('./dacon/comp1/train_gap.npy', arr= train_gap)
np.save('./dacon/comp1/test_gap.npy', arr= test_gap)
