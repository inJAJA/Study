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

# fourier
train_scr_fft = np.fft.fft(train_scr, axis = 1)/ train_scr.shape[1]
test_scr_fft = np.fft.fft(test_scr, axis = 1) / test_scr.shape[1]

train_dst_fft = np.fft.fft(train_dst, axis = 1)/ train_dst.shape[1]
test_dst_fft = np.fft.fft(test_dst, axis = 1)/ test_dst.shape[1]

# fill = 0
train_scr_fft = np.nan_to_num(train_scr_fft, copy=False)
test_scr_fft = np.nan_to_num(test_scr_fft, copy=False)

train_dst_fft = np.nan_to_num(train_dst_fft, copy=False)
test_dst_fft = np.nan_to_num(test_dst_fft, copy=False)

# save
np.save('./dacon/comp1/data/train_scr_fft.npy', arr= train_scr_fft)
np.save('./dacon/comp1/data/test_scr_fft.npy', arr= test_scr_fft)

np.save('./dacon/comp1/data/train_dst_fft.npy', arr= train_dst_fft)
np.save('./dacon/comp1/data/test_dst_fft.npy', arr= test_dst_fft)

plt.subplot(2, 1, 1)
plt.plot(train_scr[0, :], label = 'origin')
plt.xlabel('Time')
plt.ylabel('ratio')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(train_scr_fft[0, :], label = 'train_scr_fft[0]')
plt.show()