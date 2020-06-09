import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from keras.layers.merge import concatenate  

#1. data
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col= 0 , header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= None , header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col= 0 , header = 0)

print('train.shape: ', train.shape)              # (10000, 75)  = x_train, test
print('test.shape: ', test.shape)                # (10000, 71)  = x_predict
print('submission.shape: ', submission.shape)    # (10000, 4)   = y_predict

print(train.isnull().sum())                      


train = train.interpolate()                       
print(train.isnull().sum())                       

test = test.interpolate()

x = train.iloc[:, :71]                           
y = train.iloc[:, -4:]
print(x.shape)                                   # (10000, 71)
print(y.shape)                                   # (10000, 4)



x = x.fillna(x.mean())
test = test.fillna(test.mean())

# print(x.info())
# print(test.info())

x = x.values
y = y.values
x_pred = test.drop('id', axis = 1)
x_pred = x_pred.values

# x1 = x[:, 0].reshape(x.shape[0], 1)
x1 = x[:, :37]
# x2 = x[:, 1:37]
x3 = x[:, 37:]
print(x1.shape)              # (10000,)
# print(x2.shape)              # (10000, 36)
print(x3.shape)              # (10000, 34)


# x_pred1 = x_pred[:, 0].reshape(x_pred.shape[0], 1)
x_pred1 = x_pred[:, :37]
# x_pred2 = x_pred[:, 1:37]
x_pred3 = x_pred[:, 37:]

# scaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = RobustScaler()
scaler.fit(x1)
x1 = scaler.transform(x1)
x_pred1 = scaler.transform(x_pred1)

# scaler.fit(x2)
# x2 = scaler.transform(x2)
# x_pred2 = scaler.transform(x_pred2)

scaler.fit(x3)
x3 = scaler.transform(x3)
x_pred3 = scaler.transform(x_pred3)


# PCA
# pca = PCA(n_components = 10)
# pca.fit(x)
# x = pca.transform(x)
# x_pred = pca.transform(x_pred)


# train, test
x1_train, x1_test, y_train, y_test = train_test_split(x1, y, train_size = 0.8, random_state = 30)

# x2_train, x2_test, x3_train, x3_test = train_test_split(x2, x3, train_size = 0.8, random_state = 30)
x3_train, x3_test = train_test_split(x3, train_size = 0.8, random_state = 30)


#2. model
input1 = Input(shape = (37, ))
x1 = Dense(80, activation = 'elu')(input1)
x1 = Dropout(0.2)(x1)
x1 = Dense(120, activation = 'elu')(x1)
x1 = Dropout(0.2)(x1)

# input2 = Input(shape = (36, ))
# x2 = Dense(100, activation = 'elu')(input2)
# x2 = Dropout(0.2)(x2)
# x2 = Dense(120, activation = 'elu')(x2)
# x2 = Dropout(0.2)(x2)

input3 = Input(shape = (34, ))
x3 = Dense(150, activation = 'elu')(input3)
x3= Dropout(0.2)(x3)
x3 = Dense(100, activation = 'elu')(x3)
x3= Dropout(0.2)(x3)

# merge = concatenate([x1, x2, x3])
merge = concatenate([x1, x3])
middle = Dense(80, activation = 'elu')(merge)
middle = Dropout(0.2)(middle)
middle = Dense(50, activation = 'elu')(middle)
middle = Dropout(0.2)(middle)


outputs = Dense(4, activation = 'elu')(middle)
# model = Model(inputs = [input1, input2, input3], outputs = outputs)
model = Model(inputs = [input1, input3], outputs = outputs)



# earlystopping
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 50) 

#3. compile, fit
model.compile(loss = 'mae', optimizer = 'adam', metrics = ['mae'])
# model.fit([x1_train, x2_train, x3_train], y_train, epochs = 500, batch_size = 64, verbose = 2,
#          validation_split = 0.2, callbacks = [es])
model.fit([x1_train, x3_train], y_train, epochs = 500, batch_size = 64, verbose = 2,
         validation_split = 0.2, callbacks = [es])

#4. evaluate, predict
# loss_mae = model.evaluate([x1_test, x2_test, x3_test],y_test, batch_size = 64)
loss_mae = model.evaluate([x1_test, x3_test],y_test, batch_size = 64)
print('loss_mae: ', loss_mae)

# y_pred = model.predict([x_pred1, x_pred2, x_pred3])
y_pred = model.predict([x_pred1,  x_pred3])
print('y_pred: ', y_pred)


a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./dacon/comp1/y_pred6.csv', 
              index = True, header=['hhb','hbo2','ca','na'],index_label='id')

# sibmit파일
# y_pred.to_csv(경로)

'''
y_pred4 > fiilna(.mean())
loss_mae:  [1.7042060174942018, 1.7042056322097778]
y_pred:  [[ 7.5994396  4.1149335  7.530348   3.1847575]
 [ 2.802721   3.8907096  8.441036   3.177924 ]
 [ 9.247631   3.977002  10.134867   3.2490606]
 ...
 [ 9.600287   4.104522   9.561552   2.9032116]
 [ 7.7655177  3.8562138  7.9755325  2.8738363]
 [ 6.9554534  4.049231   9.3153105  3.4014254]]

y_pred5> replace(0, np.nan)
loss_mae:  [1.6567197589874267, 1.65671968460083]
y_pred:  [[7.4663086 3.926025  8.890444  3.0593545]
 [7.8958836 3.938304  8.7164345 2.9321501]
 [9.946594  4.2685084 9.891247  3.714861 ]
 ...
 [7.6045685 3.950648  8.90891   3.2338252]
 [7.6376286 3.897931  8.427586  2.8123693]
 [7.499691  3.9176478 8.738676  3.093008 ]]
'''
