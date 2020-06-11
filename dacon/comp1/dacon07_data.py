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
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= 0, header = 0)
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
print('scr:', train_scr.shape)            # scr: (10000, 35)
print('dst:', train_dst.shape)            # dst: (10000, 35)

test_scr = test.filter(regex='_src$',axis=1)     
test_dst = test.filter(regex='_dst$',axis=1)
print('scr:', test_scr.shape)             # scr: (10000, 35)
print('dst:', test_dst.shape)             # dst: (10000, 35)

# print(test_scr)


train_scr = train_scr.values                 # numpy형식으로 변경
train_dst = train_dst.values
test_scr = test_scr.values
test_dst = test_dst.values

train_gap = train_scr - train_dst             # gap = scr -dst
test_gap = test_scr - test_dst     
print(train_gap)

# 'rho'
train_rho = train.iloc[:, 0]
test_rho = test.iloc[:, 0]
print(train_rho.shape)                        # (10000,)

train_rho = train_rho.values.reshape(train_rho.shape[0], 1)   # 붙는 두개의 차원이 같아야함                  # numpy형식으로 변환
test_rho = test_rho.values.reshape(test_rho.shape[0], 1)

y = train.iloc[:, -4:]                          # train데이터에서 y값을 설정해줌                                                                  
y = y.values
'''
# print(x.info())
# print(test.info())
x= np.hstack([train_rho, train_gap])            # rho열과 gap열을 붙여줌 / 행으로 붙일 시 : np.vstack
y = y.values
x_pred = np.hstack([test_rho, test_gap])       # np.concatenate(, axis = 0)

print(x.shape)                                  # (10000, 36)
print(y.shape)                                  # (10000, 4)
print(x_pred.shape)                             # (10000, 36)



# scaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)


# train, test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 30)


#2. model
model = Sequential()
model.add(Dense(50, input_shape = (36, ), activation = 'elu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'elu'))
model.add(Dropout(0.3))
model.add(Dense(150, activation = 'elu'))
model.add(Dropout(0.3))
model.add(Dense(200, activation = 'elu'))
model.add(Dropout(0.3))
model.add(Dense(180, activation = 'elu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation = 'elu'))
model.add(Dropout(0.3))
model.add(Dense(80, activation = 'elu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation = 'elu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation = 'elu'))


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


a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./dacon/comp1/y_pred6.csv', 
              index = True, header=['hhb','hbo2','ca','na'],index_label='id')

# sibmit파일
# y_pred.to_csv(경로)
'''

x1 = train_rho
x3 = train_gap

x_pred1 = test_rho
x_pred3 = test_gap

# scaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = RobustScaler()
scaler.fit(x1)
x1 = scaler.transform(x1)
x_pred1 = scaler.transform(x_pred1)


scaler.fit(x3)
x3 = scaler.transform(x3)
x_pred3 = scaler.transform(x_pred3)

x3 = x3.reshape(-1, 35, 1)
x_pred3 = x_pred3.reshape(-1, 35, 1)

# PCA
# pca = PCA(n_components = 10)
# pca.fit(x)
# x = pca.transform(x)
# x_pred = pca.transform(x_pred)


# train, test
x1_train, x1_test, y_train, y_test = train_test_split(x1, y, train_size = 0.8, random_state = 30)

x3_train, x3_test = train_test_split(x3, train_size = 0.8, random_state = 30)


#2. model
input1 = Input(shape = (1, ))
x1 = Dense(80, activation = 'elu')(input1)
x1 = Dropout(0.2)(x1)
x1 = Dense(120, activation = 'elu')(x1)
x1 = Dropout(0.2)(x1)

input3 = Input(shape = (35, 1))
x3 = LSTM(150, activation = 'elu')(input3)
x3= Dropout(0.2)(x3)
x3 = Dense(100, activation = 'elu')(x3)
x3= Dropout(0.2)(x3)

merge = concatenate([x1, x3])
middle = Dense(80, activation = 'elu')(merge)
middle = Dropout(0.2)(middle)
middle = Dense(50, activation = 'elu')(middle)
middle = Dropout(0.2)(middle)


outputs = Dense(4, activation = 'elu')(middle)
model = Model(inputs = [input1, input3], outputs = outputs)



# earlystopping
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 50) 

#3. compile, fit
model.compile(loss = 'mae', optimizer = 'adam', metrics = ['mae'])
model.fit([x1_train, x3_train], y_train, epochs = 500, batch_size = 64, verbose = 2,
         validation_split = 0.2, callbacks = [es])

#4. evaluate, predict
loss_mae = model.evaluate([x1_test, x3_test],y_test, batch_size = 64)
print('loss_mae: ', loss_mae)

y_pred = model.predict([x_pred1,  x_pred3])
print('y_pred: ', y_pred)


a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./dacon/comp1/y_pred6.csv', 
              index = True, header=['hhb','hbo2','ca','na'],index_label='id')

