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


#2. model01
from tensorflow import keras
input_ = keras.layers.Input(shape = x_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation = 'relu')(input)
hidden2 = keras.layers.Dense(30, activation = 'relu')(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs = [input_], outputs=[output])

#2. model02
input_A = keras.layers.Input(shape =[5], name = 'wide_input')
input_B = keras.layers.Input(shape =[6], name = 'deep_input')
hidden1 = keras.layers.Dense(30, activaion = 'relu')(input_B)
hidden2 = keras.layers.Dense(30, activaion = 'relu')(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name = 'output')(concat)
# output = keras.layer.Dense(1, name = 'main_output')(concat)  
# aux_output = keras.layers.Dense(1, name = 'aux_output')(hidden2)            # 보조 출력 추가 : 다른 한쪽의 모델과 합쳐지지 않은 것
# model = keras.Model(inputs = [input_A, input_B], outputs = [output, aux_output])
model = keras.Model(inputs = [input_A, input_B], outputs = [output])


#3. compile, fit
model.compile(loss = 'mse', optimizer = keras.optimizers.SGD(lr=1e-3))
# model.compils(;oss = ['mse','mse'], loss_weights = [0.9, 0.1], optimizer = 'sgd')

x_train_A, x_train_B = x_train[:, :5], x_train[:, 2:]
x_val_A, x_val_B = x_val[:, :5], x_val[:, 2:]
x_test_A, x_test_B = x_test[:, :5], x_test[:, 2:]
x_new_A, x_new_B = x_test_A[:3], x_new_B[:3]

history = model.fit((x_train_A, x_train_B), y_train, epochs = 20, 
                    validation_data = ((x_val_A, x_val_B), y_val))
# history = model.fit([x_train_A, x_train_B], [y_train, y_train], epochs = 20, 
#                     validation_data = ([x_val_A, x_val_B], [y_val, y_val]))

mse_test = model.evaluate((x_test_A, x_test_B), y_test)
# mse_test = model.evaluate([x_test_A, x_test_B], [y_test, y_test])

y_pred= model.predict((x_new_A, x_new_B))
# y_pred_main, y_pred_aux = model.predict([x_new_A, x_new_B])
