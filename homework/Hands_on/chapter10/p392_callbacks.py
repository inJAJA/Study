from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#1. data
housing = fetch_california_housing()

x_train_full, x_test, y_train_full, y_test = train_test_split( housing.data, housing.target)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


#2. model
from sklearn import keras
model = keras.models.Sequential([
    keras.layers.Dense(30, activation = 'relu', input_shape = x_train.shape[1:]),
    keras.layers.Dense(1)
])

#3. compile, fit
model.compile(loss= 'mean_squared_error', optimizer = 'sgd')

# callbacks - ModelCehckpoint
checkpoint_cb = keras.callbacks.ModelCehckpoint('my_keras_model.h5', 
                                                save_best_only = True)
# EarlyStopping
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10,
                                                  restore_best_weights = True)

# 사용자 정의 callbacks
class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):        
    # on_train_begin(), on_train_end(), on_epoch_begin(), on_epoch_end(), on_batch_begin(), on_batch_end()
        print('\nval/train: {:2f}'.format(logs['val_loss'] / logs['loss']))

history = model.fit(x_train, y_train, epochs = 10, 
                    validation_data = [x_val, y_val],
                    callbacks = [checkpoint_cb, early_stopping_cb])
                    
model = keras.models.load_model('my_keras_model.h5')
