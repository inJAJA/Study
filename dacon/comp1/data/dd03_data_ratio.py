import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1. data
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col= 0 , header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= 0 , header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col= 0 , header = 0)

# load
train_scr = np.load('./dacon/comp1/data/train_scr.npy')
test_scr = np.load('./dacon/comp1/data/test_scr.npy')

train_dst = np.load('./dacon/comp1/data/train_dst.npy')
test_dst = np.load('./dacon/comp1/data/test_dst.npy')


# ratio
train_ratio = np.float32(train_dst/ train_scr)
test_ratio = np.float32(test_dst/ test_scr)

# 
# train_ratio = train_ratio[np.isnan(train_ratio)] = 0
train_ratio = np.nan_to_num(train_ratio, copy=False)
test_ratio = np.nan_to_num(test_ratio, copy=False)

np.save('./dacon/comp1/data/train_ratio.npy', arr= train_ratio)
np.save('./dacon/comp1/data/test_ratio.npy', arr= test_ratio)

plt.plot(train_ratio[0, :], label = 'train_ratio[0]')
plt.xlabel('Time')
plt.ylabel('ratio')
plt.legend()
plt.show()

