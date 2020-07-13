import os
import json
import numpy as np
from tqdm import tqdm
# import jovian
import numpy as np

def kaeri_metric(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: KAERI metric
    '''
    
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)
                # X, Y                     # M, V

### E1과 E2는 아래에 정의됨 ###

def E1(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: distance error normalized with 2e+04
    '''
    
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]           # X, Y
    
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)
                                            # 각 컬럼에 대한 np.sum

def E2(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: sum of mass and velocity's mean squared percentage error
    '''
    
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]             # M, V
    
    
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06))))
                                                            # 각 컬럼에 대한 np.sum
# import matplotlib as plt
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten,MaxPooling2D,BatchNormalization,Lambda
from keras.layers import AveragePooling2D, GlobalMaxPooling2D
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
import pandas as pd
from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2)
leaky.__name__ = 'leaky'

# X_data = []
# Y_data = []

X_data = np.loadtxt('./data/dacon/comp3/train_features.csv',skiprows=1,delimiter=',')
X_data = X_data[:,1:]
print(X_data.shape)
     
    
Y_data = np.loadtxt('./data/dacon/comp3/train_target.csv',skiprows=1,delimiter=',')
Y_data = Y_data[:,1:]   # ID 열 제거
print(Y_data.shape)

X_data = X_data.reshape((2800,375,5,1))
print(X_data.shape)

X_data_test = np.loadtxt('./data/dacon/comp3/test_features.csv',skiprows=1,delimiter=',')
X_data_test = X_data_test[:,1:]
X_data_test = X_data_test.reshape((700,375,5,1))

data_id = 2

plt.figure(figsize=(8,6))


plt.plot(X_data[data_id,:,0,0], label="Sensor #1")
plt.plot(X_data[data_id,:,1,0], label="Sensor #2")
plt.plot(X_data[data_id,:,2,0], label="Sensor #3")
plt.plot(X_data[data_id,:,3,0], label="Sensor #4")

plt.xlabel("Time", labelpad=10, size=20)
plt.ylabel("Acceleration", labelpad=10, size=20)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlim(0, 400)
plt.legend(loc=1)

# plt.show()

from sklearn.model_selection import train_test_split
from keras.regularizers import l1, l2, l1_l2


X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.01)
# X_train = X_data
# Y_train = Y_data
print('x_train :',X_train.shape)

weight1 = np.array([1,1,0,0])
weight2 = np.array([0,0,1,1])


def my_loss(y_true, y_pred):  # loss function은 오직 2개의 인자만 받아들임 : y_true, y_pred
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+0.000001)]) # [(y_pred-y_true),(y_true+0.000001)]을 대입
    return K.mean(K.square(divResult))


def my_loss_E1(y_true, y_pred):
    return K.mean(K.square(y_true-y_pred)*weight1)/2e+04

def my_loss_E2(y_true, y_pred):                              # 나누기 연산이 있을 때 분모가 0이 되는 것을 방지하기 위해서 더해줌
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+0.000001)])                 # 주로 K.epsilon를 더해준다.
    return K.mean(K.square(divResult)*weight2)


# tr_target = 2 

def set_model(train_target):  # 0:x,y, 1:m, 2:v
    
    activation = 'elu'
    activation2 = 'elu'
    padding = 'valid'
    model = Sequential()
    nf = 19
    fs = (5,1)
    mom = 0.991

    model.add(Conv2D(nf,fs, padding=padding,input_shape=(375,5,1), activation = activation))
    model.add(BatchNormalization(momentum = mom))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(nf*2,fs, padding=padding, activation = activation))
    model.add(BatchNormalization(momentum = mom))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(nf*4,fs, padding=padding, activation = activation))
    model.add(BatchNormalization(momentum = mom))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(nf*8,fs, padding=padding, activation = activation))
    model.add(BatchNormalization(momentum = mom))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(nf*16,fs, padding=padding, activation = activation))
    model.add(BatchNormalization(momentum = mom))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(nf*32,fs, padding=padding, activation = activation))
    model.add(BatchNormalization(momentum = mom))
    model.add(MaxPooling2D(pool_size=(2, 1)))


    model.add(Flatten())
    model.add(Dense(256, activation =activation2))
    model.add(Dense(128, activation =activation2))
    model.add(Dense(64, activation =activation2))
    model.add(Dense(32, activation =activation2))
    model.add(Dense(16, activation =activation2))
    model.add(Dense(8, activation =activation2))

    model.add(Dense(4))

    optimizer = keras.optimizers.Adam(lr = 0.001) #default = 0.001

    global weight2
    if train_target == 1: # only for M
        weight2 = np.array([0,0,1,0])
    else: # only for V
        weight2 = np.array([0,0,0,1])
       
    if train_target==0:
        model.compile(loss=my_loss_E1,
                  optimizer=optimizer,
                 )
    else:
        model.compile(loss=my_loss_E2,
                  optimizer=optimizer,
                 )
       
    model.summary()

    return model


def train(model,X,Y):
    MODEL_SAVE_FOLDER_PATH = './dacon/comp3/checkpoint/'
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)

    model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'
    best_save = ModelCheckpoint('best_m.hdf5', save_best_only=True, monitor='val_loss', mode='min')

    es = EarlyStopping(monitor = 'val_loss', verbose = 1, patience = 30)

    history = model.fit(X, Y,
                  epochs = 125,  
                  batch_size=128,
                  shuffle=True,
                  validation_split=0.2,
                  verbose = 2,
                  callbacks=[best_save])

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(history.history['loss'], 'y', label='train loss')
    loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')
    # plt.show()    
    
    return model

def plot_error(type_id,pred,true):
    print(pred.shape)

    if type_id == 0:
        _name = 'x_pos'
    elif type_id == 1:
        _name = 'y_pos'
    elif type_id == 2:
        _name = 'mass'
    elif type_id == 3:
        _name = 'velocity'
    elif type_id == 4:
        _name = "distance"
    else:
        _name = 'error'

    x_coord = np.arange(1,pred.shape[0]+1,1)
    if type_id < 2:
        Err_m = (pred[:,type_id] - true[:,type_id])
    elif type_id < 4:
        Err_m = ((pred[:,type_id] - true[:,type_id])/true[:,type_id])*100
    else:
        Err_m = ((pred[:,0]-true[:,0])**2+(pred[:,1]-true[:,1])**2)**0.5


    fig = plt.figure(figsize=(8,6))
    # plt.rcParams["font.family"]="Times New Roman"
    plt.rcParams["font.size"]=15
    plt.scatter(x_coord, Err_m, marker='o')
    plt.title("%s Prediction for Training Data" % _name, size=20)
    plt.xlabel("Data ID", labelpad=10, size=20)
    plt.ylabel("Prediction Error of %s," % _name, labelpad=10, size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.ylim(-100., 100.)
    plt.xlim(0, pred.shape[0]+1)

    # plt.show()
    
    print(np.std(Err_m))
    print(np.max(Err_m))
    print(np.min(Err_m))
    return Err_m

#  plot_error(type_id,pred,true):

def load_best_model(train_target):
    
    if train_target == 0:
        model = load_model('best_m.hdf5' , custom_objects={'my_loss_E1': my_loss, })
    else:
        model = load_model('best_m.hdf5' , custom_objects={'my_loss_E2': my_loss, })

    score = model.evaluate(X_data, Y_data, verbose=0)
    print('loss:', score)

    pred = model.predict(X_data)

    i=0

    print('정답(original):', Y_data[i])
    print('예측값(original):', pred[i])

    print('E1 :',E1(pred, Y_data))
    print('E2 :',E2(pred, Y_data))
    # print(E2M(pred, Y_data))
    # print(E2V(pred, Y_data))    
    
    if train_target ==0:
        plot_error(4,pred,Y_data)
    elif train_target ==1:
        plot_error(2,pred,Y_data)
    elif train_target ==2:
        plot_error(3,pred,Y_data)    
    
    return model

submit = pd.read_csv('./data/dacon/comp3/sample_submission.csv')

for train_target in range(3):
    model = set_model(train_target)
    train(model,X_train, Y_train)    
    best_model = load_best_model(train_target)

   
    pred_data_test = best_model.predict(X_data_test)
    
    
    if train_target == 0: # x,y 
        submit.iloc[:,1] = pred_data_test[:,0]
        submit.iloc[:,2] = pred_data_test[:,1]

    elif train_target == 1: # m 
        submit.iloc[:,3] = pred_data_test[:,2]

    elif train_target == 2: # v 
        submit.iloc[:,4] = pred_data_test[:,3]

submit.to_csv('./dacon/comp3/sub/comp3_submit.csv', index = False)

