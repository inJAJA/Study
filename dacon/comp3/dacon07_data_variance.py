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

for dataset in dataset:
    dataset = dataset.drop('time', axis = 1).values

x_id = features.reshape(-1, 375, 4)
x_pred_id = features.reshape(-1, 375, 4)

def mean(dataset):
    s1_mean, s2_mean, s3_mean, s4_mean = [],[],[],[]
    for i in range(len(x_id)):
        s1_mean = 


