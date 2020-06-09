import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.wrappers.scikit_learn import KerasRegressor
from keras.losses import MeanAbsoluteError

from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2)

#1. data
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col= 0 , header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= None , header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col= 0 , header = 0)

print('train.shape: ', train.shape)              # (10000, 75)  = x_train, test
print('test.shape: ', test.shape)                # (10000, 71)  = x_predict
print('submission.shape: ', submission.shape)    # (10000, 4)   = y_predict

print(train.isnull().sum())                      # train에 있는 null값의 합
# rho           0
# 650_src       0
# 660_src       0
# 670_src       0
# 680_src       0
#            ...
# 990_dst    1987
# hhb           0
# hbo2          0
# ca            0
# na            0
# Length: 75, dtype: int64

train = train.interpolate()                       # 보간법 : 선형보간 / 모델을 돌려서 예측 값을 넣음 / 맨 앞행은 안 채워짐
print(train.isnull().sum())                       #        : 구간을 잘라서 선에 맞게 빈자리를 채워줌
# rho        0                                    # column별 보관 : 옆의 column에 영향 X
# 650_src    0
# 660_src    0
# 670_src    0
# 680_src    0
#           ..
# 990_dst    0
# hhb        0
# hbo2       0
# ca         0
# na         0
# Length: 75, dtype: int64

test = test.interpolate()

x = train.iloc[:, :71]                           
y = train.iloc[:, -4:]
print(x.shape)                                   # (10000, 71)
print(y.shape)                                   # (10000, 4)

x = x.fillna(method = 'bfill')
test = test.fillna(method = 'bfill')

x = x.values
y = y.values
x_pred = test.drop('id', axis = 1)
x_pre = x_pred.values


# scaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)

# train, test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 30)

#2. model
def build_model(drop=0.5, optimizer = 'adam', act = 'relu'):
    inputs = Input(shape= (71, ))
    x = Dense(51, activation =act)(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = act)(x)
    x = Dropout(drop)(x)
    outputs = Dense(4, activation = act)(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['mae'],
                  loss = 'mae')
    return model


def create_hyperparameter():
    batches = [16, 32, 64, 128]
    epochs = [50, 100, 150, 200]
    dropout = np.linspace(0.1, 0.5, 5)
    activation= ['relu', 'elu', leaky]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    return {'batch_size': batches, 'epochs':epochs, 'act': activation,
            'optimizer': optimizers}

# wrapper    
model = KerasRegressor(build_fn = build_model, verbose =2)

parameter = create_hyperparameter()

# Search
search = RandomizedSearchCV(model, parameter, cv = 3)

# fit
search.fit(x_train, y_train)

#4. evaluate, predict
print(search.best_params_)

y_pred = search.predict(x_pred)
print('y_pred: ', y_pred)

from sklearn.metrics import mean_absolute_error 
mae = mean_absolute_error(x_test, y_test)
print('mae: ', mae)

y_pred = pd.DataFrame({
  'id' : test['id'],
  'hhb': y_pred[:,0],
  'hbo2': y_pred[:, 1],
  'ca': y_pred[:, 2],
  'na':y_pred[:, 3]
})
y_pred.to_csv('./dacon/y_pred2.csv', index = False )
# sibmit파일
# y_pred.to_csv(경로)
