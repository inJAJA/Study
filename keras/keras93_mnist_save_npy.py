import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist                         
mnist.load_data()                                         

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
print(x_train[0])                                         
print('y_train: ' , y_train[0])                           # 5

print(x_train.shape)                                      # (60000, 28, 28)
print(x_test.shape)                                       # (10000, 28, 28)
print(y_train.shape)                                      # (60000,)       
print(y_test.shape)                                       # (10000,)



""" numpy로 저장 """
np.save('./data/mnist_train_x.npy', arr = x_train)
np.save('./data/mnist_train_y.npy', arr = y_train)
np.save('./data/mnist_test_x.npy', arr = x_test)
np.save('./data/mnist_test_y.npy', arr = y_test)
# np.save( '------경로----------', arr = 저장할 내용)

# 데이터 전처리 1. 원핫인코딩 : 당연하다             
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                                      #  (60000, 10)

# 데이터 전처리 2. 정규화( MinMaxScalar )                                                    
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') /255  
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') /255.                                     

