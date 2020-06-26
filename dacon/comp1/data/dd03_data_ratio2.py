import numpy as np
import pandas as pd


train_scr = np.load('./dacon/comp1/data/train_scr_fft2.npy')
test_scr = np.load('./dacon/comp1/data/test_scr_fft2.npy')

train_dst = np.load('./dacon/comp1/data/train_dst_fft2.npy')
test_dst = np.load('./dacon/comp1/data/test_dst_fft2.npy')

# 제곱
train_scr2 = np.square(train_scr)
test_scr2 = np.square(test_scr)

train_dst2 = np.square(train_dst)
test_dst2 = np.square(test_dst)

#ratio
train_ratio2 = np.float64(train_scr / train_dst)
test_ratio2 = np.float64(test_scr / test_dst)

# NaN제거
train_ratio2 = np.nan_to_num(train_ratio2, copy=False)
test_ratio2 = np.nan_to_num(test_ratio2, copy=False)
print(train_ratio2[20, :])


np.save('./dacon/comp1/data/train_ratio2.npy', arr= train_ratio2)
np.save('./dacon/comp1/data/test_ratio2.npy', arr= test_ratio2)