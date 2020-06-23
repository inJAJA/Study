import numpy as np
import pandas as pd

#1. data
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col= 0 , header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= 0 , header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col= 0 , header = 0)

print('train.shape: ', train.shape)              # (10000, 75)  = x_train, test
print('test.shape: ', test.shape)                # (10000, 71)  = x_predict
print('submission.shape: ', submission.shape)    # (10000, 4)   = y_predict

# rho
train_rho = train.iloc[:, 0]   
test_rho = test.iloc[:, 0]  

train_rho = train_rho.interpolate(axis = 0)
test_rho= test_rho.interpolate(axis = 0)

train_rho = train_rho.fillna(train_rho.mean())
test_rho = test_rho.fillna(test_rho.mean())
  
print(train_rho.shape)                        # (10000, )
print(test_rho.shape)                         # (10000, )


train_rho = train_rho.values.reshape(-1, 1)
test_rho = test_rho.values.reshape(-1, 1)

train_rho2 = np.square(train_rho)
test_rho2 = np.square(test_rho)

np.save('./dacon/comp1/data/train_rho2.npy', arr= train_rho2)
np.save('./dacon/comp1/data/test_rho2.npy', arr= test_rho2)
