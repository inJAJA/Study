# RandomiziedSearchCV
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical

from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2)

#1. data
iris = load_iris()

x = iris.data
y = iris.target

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, 
                                                    shuffle = True, random_state = 43)


# gridsearch에 넣기위한 모델(모델에 대한 명시 : 함수로 만듦)
def build_model(drop=0.5, optimizer = 'adam', act = 'relu'):
    inputs = Input(shape= (4, ), name = 'input')
    x = Dense(51, activation = act, name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = act, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = act, name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation = 'softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    return model

# parameter
def create_hyperparameters(): # epochs, node, acivation 추가 가능
    batches = [64, 128, 512]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5).tolist()           
    activation = ['relu', 'elu', leaky]               
    return {'deep__batch_size' : batches, 'deep__optimizer': optimizers, 'deep__act':activation,
           'deep__drop': dropout}                                       

# wrapper
model = KerasClassifier(build_fn = build_model, verbose = 1)

hyperparameters = create_hyperparameters()

from sklearn.pipeline import Pipeline, make_pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('deep', model)])
# pipe = make_pipeline(StandardScaler(), model)                        # kerasclassifier__ :

# gridsearch
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
search = RandomizedSearchCV(pipe, hyperparameters, cv = 3)                        


# fit
search.fit(x_train, y_train)

print(search.best_params_)  

score = search.score(x_test, y_test)
print('acc: ', score)



