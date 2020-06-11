# 시계열에서 시작 시간이 맞지 않을 경우 '0'으로 채운다.
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline

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

        print(i,':',end)
        print(sum1)
        print('=====')
        s1.append(sum1)
        s2.append(sum2)
        s3.append(sum3)
        s4.append(sum4)

    return np.array([s1, s2, s3, s4])

train_time = isnull_data(features)
test_time = isnull_data(test)






