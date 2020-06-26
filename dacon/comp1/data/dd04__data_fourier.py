import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load
train_scr = np.load('./dacon/comp1/data/train_scr.npy')
test_scr = np.load('./dacon/comp1/data/test_scr.npy')

train_dst = np.load('./dacon/comp1/data/train_dst.npy')
test_dst = np.load('./dacon/comp1/data/test_dst.npy')

# fourier
n1_1 = train_scr.shape[1]
n1_2 = test_scr.shape[1]

n2_1 = train_dst.shape[1]
n2_2 = test_dst.shape[1]

# 실수부
train_scr_fft = np.fft.fft((train_scr - train_scr.mean() ), axis = 1).real/ n1_1
test_scr_fft = np.fft.fft((test_scr - test_scr.mean()), axis = 1).real / n1_2

train_dst_fft = np.fft.fft((train_dst - train_dst.mean()), axis = 1).real/ n2_1
test_dst_fft = np.fft.fft((test_dst - test_dst.mean()), axis = 1).real/ n2_2

# 허수부
train_scr_fft_imag = np.fft.fft((train_scr - train_scr.mean()), axis = 1).imag/ n1_1
test_scr_fft_imag = np.fft.fft((test_scr - test_scr.mean()), axis = 1).imag / n1_2

train_dst_fft_imag = np.fft.fft((train_dst - train_dst.mean()), axis = 1).imag/ n2_1
test_dst_fft_imag = np.fft.fft((test_dst - test_dst.mean()), axis = 1).imag/ n2_2

# half
train_scr_fft = train_scr_fft[:, range(int(n1_1/2)) ]
test_scr_fft = test_scr_fft[:, range(int(n1_2/2)) ]

train_dst_fft = train_dst_fft[:, range(int(n2_1/2)) ]
test_dst_fft = test_dst_fft[:, range(int(n2_2/2)) ]

# half
train_scr_fft_imag = train_scr_fft_imag[:, range(int(n1_1/2)) ]
test_scr_fft_imag = test_scr_fft_imag[:, range(int(n1_2/2)) ]

train_dst_fft_imag = train_dst_fft_imag[:, range(int(n2_1/2)) ]
test_dst_fft_imag = test_dst_fft_imag[:, range(int(n2_2/2)) ]


plt.plot(train_scr[0,:], label = 'train_scr_fft0')
plt.plot(train_scr_fft[0,:], label = 'train_scr_fft0')
# plt.plot(train_scr_fft[5,:], label = 'train_dst_fft5')
# plt.plot(train_scr_fft_imag[5,:], label = 'train_scr_fft5')
plt.legend()
plt.show()

# # abs
# train_scr_fft = np.abs(train_scr_fft)
# test_scr_fft = np.abs(test_scr_fft)

# train_dst_fft = np.abs(train_dst_fft)
# test_dst_fft = np.abs(test_dst_fft)

# save
np.save('./dacon/comp1/data/train_scr_fft.npy', arr= train_scr_fft)
np.save('./dacon/comp1/data/test_scr_fft.npy', arr= test_scr_fft)

np.save('./dacon/comp1/data/train_dst_fft.npy', arr= train_dst_fft)
np.save('./dacon/comp1/data/test_dst_fft.npy', arr= test_dst_fft)


np.save('./dacon/comp1/data/train_scr_fft_imag.npy', arr= train_scr_fft_imag)
np.save('./dacon/comp1/data/test_scr_fft_imag.npy', arr= test_scr_fft_imag)

np.save('./dacon/comp1/data/train_dst_fft_imag.npy', arr= train_dst_fft_imag)
np.save('./dacon/comp1/data/test_dst_fft_imag.npy', arr= test_dst_fft_imag)

print(train_scr_fft.shape)
print(test_scr_fft.shape)

print(train_dst_fft.shape)
print(test_dst_fft.shape)


