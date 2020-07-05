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
history = model.fit(x_train, y_train, epochs = 20, validation_data = (x_val, y_val))

#4. evaluate, predict
mse_test = model.evaluate(x_test, y_test)

x_new = x_test[:3]
y_pred = model.predict(x_new)