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

# load
train_scr = np.load('./dacon/comp1/data/train_scr.npy')
test_scr = np.load('./dacon/comp1/data/test_scr.npy')

train_dst = np.load('./dacon/comp1/data/train_dst.npy')
test_dst = np.load('./dacon/comp1/data/test_dst.npy')

# gap
train_gap = train_scr - train_dst                  # gap = scr -dst
test_gap = test_scr - test_dst     
print(train_gap)

# nan = 0
train_gap = np.nan_to_num(train_gap, copy=False)
test_gap = np.nan_to_num(test_gap, copy=False)
print(train_gap)


np.save('./dacon/comp1/data/train_gap.npy', arr= train_gap)
np.save('./dacon/comp1/data/test_gap.npy', arr= test_gap)
