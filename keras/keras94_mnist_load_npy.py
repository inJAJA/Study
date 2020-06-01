import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist                         

x_train = np.load('./data/mnist_train_x.npy')
y_train = np.load('./data/mnist_train_y.npy')
x_test = np.load('./data/mnist_test_x.npy')
y_test = np.load('./data/mnist_test_y.npy')





# 데이터 전처리 1. 원핫인코딩 : 당연하다             
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                                      #  (60000, 10)

# 데이터 전처리 2. 정규화( MinMaxScalar )                                                    
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') /255  
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') /255.                                     


#2.3. 모델 불러오기
from keras.models import load_model                     # (save_wights_only = False)
model = load_model('./model/check-08-0.0540.hdf5')      # model과 weight가 같이 저장되어 있음 
                                                        # model, compile, fit부분이 필요없다.


#4. 평가
loss_acc = model.evaluate(x_test, y_test, batch_size= 64)
print('loss_acc: ', loss_acc)                     

                         

'''     
loss_acc:  [0.041936698463978246, 0.9876999855041504]
'''