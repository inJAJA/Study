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

from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2)

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

# 'rho'
train_rho = train.iloc[:, 0]
test_rho = test.iloc[:, 0]
print(train_rho.shape)                             # (10000,)

train_rho = train_rho.values                
test_rho = test_rho.values
x = train.iloc[:, :-4].values
x_pred = test.values
print(x.shape)

y = train.iloc[:, -4:].values                       # train데이터에서 y값을 설정해줌    


x1 = train_rho.reshape(-1, 1)                       # scaler에 넣어주기 위해 2차원 변경
x2 = x[:, 1:36]
x3 = x[:, 36:]
x4 = train_gap

print(x1.shape)     # (10000,)
print(x2.shape)     # (10000, 35)
print(x3.shape)     # (10000, 35)
print(x4.shape)     # (10000, 35)


x_pred1 = test_rho.reshape(-1, 1)                   # scaler에 넣어주기 위해 2차원 변경
x_pred2 = x_pred[:, 1:36]
x_pred3 = x_pred[:, 36: ]
x_pred4 = test_gap

# scaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = RobustScaler()
scaler.fit(x1)
x1 = scaler.transform(x1)
x_pred1 = scaler.transform(x_pred1)

scaler.fit(x2)
x2 = scaler.transform(x2)
x_pred2 = scaler.transform(x_pred2)

scaler.fit(x3)
x3 = scaler.transform(x3)
x_pred3 = scaler.transform(x_pred3)

scaler.fit(x4)
x4 = scaler.transform(x4)
x_pred4 = scaler.transform(x_pred4)

# PCA
# pca = PCA(n_components = 10)
# pca.fit(x)
# x = pca.transform(x)
# x_pred = pca.transform(x_pred)


# train, test
x1_train, x1_test, y_train, y_test = train_test_split(x1, y, train_size = 0.8, random_state = 30)
x2_train, x2_test, x3_train, x3_test = train_test_split(x2, x3, train_size = 0.8, random_state = 30)
x4_train, x4_test = train_test_split(x4, train_size = 0.8, random_state = 30)


#2. model
input1 = Input(shape = (1, ))
x1 = Dense(80, activation = 'elu')(input1)
x1 = Dropout(0.2)(x1)
x1 = Dense(120, activation = 'elu')(x1)
x1 = Dropout(0.2)(x1)

input2 = Input(shape = (35, ))
x2 = Dense(80, activation = 'elu')(input2)
x2 = Dropout(0.2)(x2)
x2 = Dense(120, activation = 'elu')(x2)
x2 = Dropout(0.2)(x2)

input3 = Input(shape = (35, ))
x3 = Dense(80, activation = 'elu')(input3)
x3= Dropout(0.2)(x3)
x3 = Dense(120, activation = 'elu')(x3)
x3= Dropout(0.2)(x3)

input4 = Input(shape = (35, ))
x4 = Dense(150, activation = 'elu')(input4)
x4= Dropout(0.2)(x4)
x4 = Dense(100, activation = 'elu')(x4)
x4= Dropout(0.2)(x4)

# merge = concatenate([x1, x2, x3])
merge = concatenate([x1, x2, x3, x4])
middle = Dense(80, activation = 'elu')(merge)
middle = Dropout(0.2)(middle)
middle = Dense(50, activation = 'elu')(middle)
middle = Dropout(0.2)(middle)


outputs = Dense(4, activation = 'elu')(middle)
# model = Model(inputs = [input1, input2, input3], outputs = outputs)
model = Model(inputs = [input1, input2, input3, input4], outputs = outputs)



# earlystopping
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 50) 

#3. compile, fit
model.compile(loss = 'mae', optimizer = 'adam', metrics = ['mae'])
model.fit([x1_train, x2_train, x3_train, x4_train], y_train, epochs = 500, batch_size = 128, verbose = 2,
         validation_split = 0.2, callbacks = [es])

#4. evaluate, predict
loss_mae = model.evaluate([x1_test, x2_test, x3_test, x4_test],y_test, batch_size = 128)
print('loss_mae: ', loss_mae)

y_pred = model.predict([x_pred1, x_pred2, x_pred3, x_pred4])
print('y_pred: ', y_pred)


a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./dacon/comp1/y_pred7_elu.csv', 
              index = True, header=['hhb','hbo2','ca','na'],index_label='id')
'''
loss_mae:  [1.6452705392837523, 1.6452704668045044]
y_pred:  [[ 8.011511   3.9454927  8.913646   2.9070392]
 [ 8.031633   3.9876478  8.918821   2.8786464]
 [10.046907   4.1862535  9.43985    3.4965138]
 ...
 [ 7.868608   4.0444684  9.084156   3.0219097]
 [ 7.9418683  3.9243941  8.818521   2.8000116]
 [ 8.084814   3.9495373  8.780014   2.9098148]]

 loss_mae:  [1.6670129718780518, 1.667013168334961

 loss_mae:  [1.6643087844848632, 1.6643086671829224] : batch64, leaky

 loss_mae:  [1.6606090335845947, 1.6606090068817139] : batch128, leaky
'''