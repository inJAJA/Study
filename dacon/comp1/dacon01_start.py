import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA

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

# print(x.info())
# print(test.info())

x = x.values
y = y.values
x_pred = test.drop('id', axis = 1)
x_pred = x_pred.values

# scaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)

# PCA
# pca = PCA(n_components = 10)
# pca.fit(x)
# x = pca.transform(x)
# x_pred = pca.transform(x_pred)


# train, test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 30)

#2. model
model = Sequential()
model.add(Dense(50, input_shape = (71, ), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(150, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(180, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(80, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation = 'relu'))


# earlystopping
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 50) 

#3. compile, fit
model.compile(loss = 'mae', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs = 500, batch_size = 64, verbose = 2,
         validation_split = 0.2, callbacks = [es])

#4. evaluate, predict
loss_mae = model.evaluate(x_test, y_test, batch_size = 64)
print('loss_mae: ', loss_mae)

y_pred = model.predict(x_pred)
print('y_pred: ', y_pred)

y_pred = pd.DataFrame({
  'id' : test['id'],
  'hhb': y_pred[:,0],
  'hbo2': y_pred[:, 1],
  'ca': y_pred[:, 2],
  'na':y_pred[:, 3]
})
# y_pred = pd.DataFrame({
#   'id' : np.array(range(10000, 20000)),
#   'hhb': y_pred[:,0],
#   'hbo2': y_pred[:, 1],
#   'ca': y_pred[:, 2],
#   'na':y_pred[:, 3]
# })
# y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('./dacon/y_pred4.csv', index = False )

# a = np.arange(10000,20000)
# y_pred = pd.DataFrame(y_pred,a)
# y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', 
#               index = True, header=['hhb','hbo2','ca','na'],index_label='id')

# sibmit파일
# y_pred.to_csv(경로)

'''
loss_mae:  [1.6898496789932251, 1.6898494958877563] 
y_pred:  [[3.5748956 3.3686323 6.9966373 2.2605565] 
 [3.1582272 3.438317  7.171153  2.3042784]
 [5.0822406 3.4805655 7.4849043 2.3881085]
 ...
 [3.0626209 3.33351   6.7968307 2.258009 ]
 [2.1190577 4.0426445 9.141554  2.779194 ]
 [3.6889172 3.3834476 7.0326495 2.2775798]]
'''
