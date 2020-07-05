from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np

#1. data
housing = fetch_california_housing()

x_train_full, x_test, y_train_full, y_test = train_test_split( housing.data, housing.target)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


# build_model
def build_model(n_hidden = 1, n_neurons = 30, learning_rate = 3e-3, input_shape = [8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape = input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation = 'relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr = learning_rate)
    model.compile(loss='mse', optimizer = optimizer)
    return model


# KerasRegressor
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
'''
keras_reg.fit(x_train, y_train, epochs = 100,
               validation_data = (x_val, y_val),
               callbacks = [keras.callbacks.EarlyStopping(patience = 10)])

# evaluate
mse_test = keras_reg.score(x_test, y_test)

x_new = x_test[:3]
y_pred = keras_reg.predict(x_new) 
'''

# RandomizedSearchCVn
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    'n_hidden':[0, 1, 2, 3],
    'n_neurons':np.arange(1, 100),
    'learning_rate':reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter =10, cv =3)
rnd_search_cv.fit(x_train, y_train, epochs = 100,
               validation_data = (x_val, y_val),  # cross_validation을 사용함으로 x_val, y_val 사용 X
               callbacks = [keras.callbacks.EarlyStopping(patience = 10)])

print(rnd_search_cv.best_params_)
print(rnd_search_cv.best_score_)

model = rnd_search_cv.best_estimator_.model  # 최상의 훈련된 model