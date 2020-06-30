import cv2
import glob
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input, BatchNormalization
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, f1_score
from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2) 
leaky.__name__ = 'leaky'

#1. data
x_train = np.load('/tf/notebooks/Ja/data/x_train.npy').reshape(-1, 56, 56, 1)  # 32, 32
x_pred = np.load('/tf/notebooks/Ja/data/x_test.npy').reshape(-1, 56, 56, 1)
x_val = np.load('/tf/notebooks/Ja/data/x_val.npy').reshape(-1, 56, 56, 1)

# print(x_train.shape)
# print(x_pred.shape)
# print(x_val.shape)

y_train = np.load('/tf/notebooks/Ja/data/y_train.npy')
y_val = np.load('/tf/notebooks/Ja/data/y_test.npy')
# print(y_train.shape)
# print(y_val.shape)


x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size = 0.5,
                                                    shuffle= True, random_state = 66)


#2. model
input1 = Input(shape = (56, 56, 1))
x = Conv2D(50, (3, 3), padding= 'valid')(input1)
x = LeakyReLU(alpha = 0.2)(x) 
x = BatchNormalization()(x)
x = Conv2D(150, (3, 3), padding= 'valid')(x)
x = LeakyReLU(alpha = 0.2)(x) 
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size= 2)(x)
x = Conv2D(250, (3, 3), padding= 'valid')(x)
x = LeakyReLU(alpha = 0.2)(x) 
x = BatchNormalization()(x)
x = Conv2D(350, (3, 3), padding= 'valid')(x)
x = LeakyReLU(alpha = 0.2)(x) 
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size= 2)(x)
x = Conv2D(450, (3, 3), padding= 'same')(x)
x = LeakyReLU(alpha = 0.2)(x) 
x = BatchNormalization()(x)
x = Conv2D(450, (3, 3), padding= 'same')(x)
x = LeakyReLU(alpha = 0.2)(x) 
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size= 2)(x)
x = Conv2D(450, (3, 3), padding= 'same')(x)
x = LeakyReLU(alpha = 0.2)(x) 
x = BatchNormalization()(x)
x = Conv2D(450, (3, 3), padding= 'same')(x)
x = LeakyReLU(alpha = 0.2)(x) 
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size= 2)(x)
x = Conv2D(450, (3, 3), padding= 'same')(x)
x = LeakyReLU(alpha = 0.2)(x) 
x = BatchNormalization()(x)
x = Conv2D(450, (3, 3), padding= 'same')(x)
x = LeakyReLU(alpha = 0.2)(x) 
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size= 2)(x)
x = Flatten()(x)
x = Dense(600)(x)
x = LeakyReLU(alpha = 0.2)(x) 
x = Dense(300)(x)
x = LeakyReLU(alpha = 0.2)(x) 
x = Dense(100)(x)
output = Dense(1, activation = 'sigmoid')(x)
model = Model(inputs = input1, outputs = output)

model.summary()

# callbacks
es = EarlyStopping(monitor = 'loss', patience = 50, verbose = 1)

modelpath = '/tf/notebooks/Ja/save_model/model_save3_{epoch:02d} - {val_loss:.4f}.hdf5'                         
checkpoint = ModelCheckpoint(filepath= modelpath, monitor='val_loss',
                            save_best_only = True, mode = 'auto', save_weights_only= False)                

#3. compile, fit
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 300, batch_size = 128, verbose = 2,
                            callbacks = [es, checkpoint],
                            validation_data = (x_val, y_val))

# model.save
model.save('/tf/notebooks/Ja/save_model/model02_1.h5')


#4. eavluate
loss_acc = model.evaluate(x_test, y_test, batch_size = 128)
print('loss_acc : ', loss_acc)

'''
# load_model
from keras.models import load_model
model = load_model('/tf/notebooks/Ja/save_model/model02_1.h5')
'''
# f1_score
y_pred = model.predict(x_test)
y_pred = np.where(y_pred >= 0.5, 1, 0)
f1_score = f1_score(y_test, y_pred)
print('f1_score : ', f1_score)


# submit_data
y_predict = model.predict(x_pred)
y_predict = np.where(y_predict >= 0.5, 1, 0)
y_predict = y_predict.reshape(-1,)
np.save('/tf/notebooks/Ja/sub/y_predict.npy', arr = y_predict)
print('save_complete')

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
