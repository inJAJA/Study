import cv2
import glob
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, f1_score
from keras.models import load_model
from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2) 
leaky.__name__ = 'leaky'
'''
#1. data
x_train = np.load('/tf/notebooks/Ja/data/x_train.npy').reshape(-1, 56, 56, 1)
x_pred = np.load('/tf/notebooks/Ja/data/x_test.npy').reshape(-1, 56, 56, 1)
x_val = np.load('/tf/notebooks/Ja/data/x_val.npy').reshape(-1, 56, 56, 1)

y_train = np.load('/tf/notebooks/Ja/data/y_train.npy')
y_val = np.load('/tf/notebooks/Ja/data/y_test.npy')

# load_model
model = load_model('/tf/notebooks/Ja/save_model/model02_3.h5')
model.summary()
# y_pred = model.predict(x_test)
# f1_score = f1_score(y_test, y_pred)
# print('f1_score : ', f1_score)


# submit_data
y_predict = model.predict(x_pred)
y_predict = np.where(y_predict >= 0.5, 1, 0)
y_predict = y_predict.reshape(-1,)

np.save('/tf/notebooks/Ja/sub/y_predict.npy', arr = y_predict)
print('save_complete')
'''
# y_predict_load
y_predict  = np.load('/tf/notebooks/Ja/sub/y_predict.npy')
y_predict = y_predict.reshape(-1,)
print(y_predict.shape)

# submission
def submission(y_sub):
    for i in range(len(y_sub)):
        path = '/tf/notebooks/Ja/data/test/test_label.txt'
        f1 = open(path, 'r')
        title = f1.read().splitlines()
        f = open('/tf/notebooks/Ja/sub/submission1.txt', 'a', encoding='utf-8')
        f.write(title[i]+' '+str(y_sub[i]) + '\n')
    print('complete')

submission(y_predict)


