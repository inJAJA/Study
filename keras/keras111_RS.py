from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, MaxPooling2D, Dense, LSTM
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam
import numpy as np
# from keras.layers import LeakyReLU
# leaky = LeakyReLU(alpha= 0.2)
import tensorflow as tf
leaky = tf.nn.leaky_relu   # load_model에서 사용 불가

#1. data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)                                   # (60000, 28, 28)
print(x_test.shape)                                    # (10000, 28, 28)

x_train = x_train.reshape(x_train.shape[0], 28*28)/225
x_test = x_test.reshape(x_test.shape[0], 28*28)/225

# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)                                    # (60000, 10)

#2. model

# gridsearch에 넣기위한 모델(모델에 대한 명시 : 함수로 만듦)
def build_model(drop=0.5, optimizer = Adam, lr = 0.01, act = 'relu'):
    inputs = Input(shape= (28*28, ), name = 'input')
    x = Dense(51, activation = act, name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = act, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = act, name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation = 'softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)

    opt = optimizer(lr = lr)

    model.compile(optimizer = opt, metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    return model

# parameter
def create_hyperparameters(): # epochs, node, acivation 추가 가능
    batches = [384, 256]
    optimizers = [Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam]
    learning_rate = [0.05, 0.1, 0.5]
    dropout = np.linspace(0.3, 0.5, 3).tolist()      
    activation = [leaky]                     
    return {'batch_size' : batches, 'optimizer': optimizers, 'lr':learning_rate,
           'drop': dropout, 'act': activation}                                       

# wrapper
from keras.wrappers.scikit_learn import KerasClassifier          
model = KerasClassifier(build_fn = build_model, verbose = 1, epochs = 1)

hyperparameters = create_hyperparameters()

# gridsearch
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
# search = RandomizedSearchCV(model, hyperparameters, cv = 3, n_jobs = 5)          
search = RandomizedSearchCV(model, hyperparameters, cv = 3, n_iter = 1)                        

# fit
search.fit(x_train, y_train)

print(search.best_params_)  


score = search.score(x_test, y_test)
print('acc: ', score)