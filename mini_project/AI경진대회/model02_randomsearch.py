import cv2
import glob
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input, BatchNormalization
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split, RandomizedSearchCV
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, f1_score
import time
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


start = time.time()
#2. model
def build_model(drop = 0.2, opt = 'adam', pool = 2, alpha = 0.2):
    input1 = Input(shape = (56, 56, 1))
    x = Conv2D(50, (3, 3), padding= 'valid')(input1)
    x = LeakyReLU(alpha = alpha)(x) 
    x = BatchNormalization()(x)
    x = Conv2D(150, (3, 3), padding= 'valid')(x)
    x = LeakyReLU(alpha = alpha)(x) 
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size= pool)(x)
    x = Conv2D(250, (3, 3), padding= 'valid')(x)
    x = LeakyReLU(alpha = alpha)(x) 
    x = BatchNormalization()(x)
    x = Conv2D(350, (3, 3), padding= 'valid')(x)
    x = LeakyReLU(alpha = alpha)(x) 
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size= pool)(x)
    x = Conv2D(450, (3, 3), padding= 'same')(x)
    x = LeakyReLU(alpha = alpha)(x) 
    x = BatchNormalization()(x)
    x = Conv2D(450, (3, 3), padding= 'same')(x)
    x = LeakyReLU(alpha = alpha)(x) 
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size= pool)(x)
    x = Conv2D(550, (3, 3), padding= 'same')(x)
    x = LeakyReLU(alpha = alpha)(x)
    x = Dropout(drop)(x)
    x = BatchNormalization()(x)
    x = Conv2D(650, (3, 3), padding= 'same')(x)
    x = LeakyReLU(alpha = alpha)(x) 
    x = Dropout(drop)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size= pool)(x)
    x = Conv2D(450, (3, 3), padding= 'same')(x)
    x = LeakyReLU(alpha = alpha)(x) 
    x = BatchNormalization()(x)
    x = Conv2D(450, (3, 3), padding= 'same')(x)
    x = LeakyReLU(alpha = alpha)(x) 
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size= pool)(x)
    x = Flatten()(x)
    x = Dense(600)(x)
    x = LeakyReLU(alpha = alpha)(x) 
    x = Dropout(drop)(x)
    x = Dense(300)(x)
    x = LeakyReLU(alpha = alpha)(x) 
    x = Dropout(drop)(x)
    x = Dense(100)(x)
    output = Dense(1, activation = 'sigmoid')(x)
    model = Model(inputs = input1, outputs = output)

    model.summary()
               
    #3. compile, fit
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['acc'])

    return model

keras = KerasClassifier(build_fn = build_model, verbose = 2)

def create_hyperparameter():
    batches = [128, 256]
    dropout = [0, 0.1, 0.3, 0.5]
    # activation= ['relu', 'elu', 'leaky']
    pool = [2, 3]
    alpha = [0.1, 0.2]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    return {'batch_size': batches, 'drop': dropout, 'pool': pool, #'act': activation, 
            'opt': optimizers, 'alpha':alpha}

params = create_hyperparameter()

# callbacks
es = EarlyStopping(monitor = 'loss', patience = 50, verbose = 1)

modelpath = '/tf/notebooks/Ja/save_model/model_save1_1_{epoch:02d} - {val_loss:.4f}.hdf5'                         
checkpoint = ModelCheckpoint(filepath= modelpath, monitor='val_loss',
                                save_best_only = True, mode = 'auto', save_weights_only= False) 

model = RandomizedSearchCV(keras, params, cv = 3, n_iter= 10) 
                                                # n_iter : 해당 갯수만큼의 경우만 돌리겠다 * cv / default = 10
model.fit(x_train, y_train, epochs = 300,
                            callbacks = [es, checkpoint],
                            validation_data = (x_val, y_val))

print(model.best_params_)

# model.save
model.save('/tf/notebooks/Ja/save_model/model01_1.h5')

#4. eavluate
loss_acc = model.evaluate(x_val, y_val)
print('loss_acc : ', loss_acc)

# from keras.models import load_model
# model = load_model('/tf/notebooks/Ja/save_model/model01_1.h5')

y_pred = model.predict(x_val)
y_pred = np.where(y_pred >= 0.5, 1, 0)
f1_score = f1_score(y_val, y_pred)
print('f1_score : ', f1_score)


# submit_data
y_predict = model.predict(x_pred)
y_predict = np.where(y_predict >= 0.5, 1, 0)
y_predict = y_predict.reshape(-1,)
np.save('/tf/notebooks/Ja/sub/y_predict1_1.npy', arr = y_predict)
print('save_complete')

# submission
def submission(y_sub):
    for i in range(len(y_sub)):
        path = '/tf/notebooks/Ja/data/test/test_label.txt'
        f1 = open(path, 'r')
        title = f1.read().splitlines()
        f = open('/tf/notebooks/Ja/sub/submission1_1.txt', 'a', encoding='utf-8')
        f.write(title[i]+' '+str(y_sub[i]) + '\n')
    print('complete')

submission(y_predict)


end = time.time() - start
print('END :', end )

'''
submission2_1, model02_1, model_save2_1, y_predict2_1| f1_score :  0.9999750255987613
'''