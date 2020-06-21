from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np

#1. data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)                                           # (60000, 28, 28)
print(x_test.shape)                                            # (10000, 28, 28)

x_train = x_train.reshape(x_train.shape[0], 28*28)/225
x_test = x_test.reshape(x_test.shape[0], 28*28)/225

# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)                                            # (60000, 10)

#2. model

# gridsearch에 넣기위한 모델(모델에 대한 명시 : 함수로 만듦)
def build_model(drop=0.5, optimizer = 'adam'):
    inputs = Input(shape= (28*28, ), name = 'input')
    x = Dense(51, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation = 'softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    return model

# parameter
def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5).tolist()                           # start = 0.1, end = 0.5, 5개 생성
    return{'batch_size' : batches, 'optimizer': optimizers, 
           'drop': dropout}                                      # dictionary형태

# wrapper : sklearn에서 쓸수 있도로 keras모델 wrapping
from keras.wrappers.scikit_learn import KerasClassifier          # 분류모뎅이라 Classifier
model = KerasClassifier(build_fn = build_model, verbose = 1)

hyperparameters = create_hyperparameters()

# gridsearch
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
search = GridSearchCV(model, hyperparameters, cv = 3)            # cv = cross_validation
# batches * optimizers * dropout * cv
#   5     *     3      *    5    *  3 = 225번의 model이 돌아감 

# fit
search.fit(x_train, y_train)

print(search.best_params_)   
# .best_estimator_ : 최고 점수를 낸 파라미터를 가진 모형
# .best_params_ : 최고점수를 낸 파라미터
# .best_score_ : 최고 점수