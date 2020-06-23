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


# scr
train_scr = train.filter(regex='_src$',axis=1)
test_scr = test.filter(regex='_src$',axis=1)

# dst
train_dst = train.filter(regex='_dst$',axis=1)
test_dst = test.filter(regex='_dst$',axis=1)

# rho열 제거 후 interpolate
train_scr = train_scr.interpolate(axis = 1)
test_scr= test_scr.interpolate(axis = 1)

train_dst = train_dst.interpolate(axis = 1)
test_dst= test_dst.interpolate(axis = 1)

# fillna = bfill
train_scr = train_scr.fillna(method = 'bfill', axis = 1)
test_scr = test_scr.fillna(method = 'bfill', axis = 1)

train_dst = train_dst.fillna(method = 'bfill', axis = 1)
test_dst = test_dst.fillna(method = 'bfill', axis = 1)

# fillna = mean
train_scr = train_scr.fillna(train_scr.mean(axis = 1))
test_scr = test_scr.fillna(test_scr.mean(axis = 1))

train_dst = train_dst.fillna(train_dst.mean(axis = 1))
test_dst = test_dst.fillna(test_dst.mean(axis = 1))

# numpy
train_scr = train_scr.values
test_scr = test_scr.values

train_dst = train_dst.values
test_dst = test_dst.values

# save
np.save('./dacon/comp1/data/train_scr.npy', arr= train_scr)
np.save('./dacon/comp1/data/test_scr.npy', arr= test_scr)

np.save('./dacon/comp1/data/train_dst.npy', arr= train_dst)
np.save('./dacon/comp1/data/test_dst.npy', arr= test_dst)