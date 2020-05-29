import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,  LSTM
from sklearn.datasets import load_boston
boston = load_boston()

x = boston.data
y = boston.target 
print(x.shape)           # (506, 13)
print(y.shape)           # (506,)

# minmaxscaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(x[0,-1])

x = x.reshape(x.shape[0], 13, 1)

# train_test
from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test = train_test_split(x, y, random_state = 30,
                                    test_size = 101/506)

#2. model
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (13,1)))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

model.summary()

# callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
# earlystopping
es = EarlyStopping(monitor = 'val_loss', patience = 50, verbose = 1)
# modelcheckpoint
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                    save_best_only = True)
# TensorBoard
ts_board = TensorBoard(log_dir = 'graph', histogram_freq =0, 
                       write_graph = True, write_images = True)

#3. compile, fit 
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
hist = model.fit(x_train, y_train, epochs =300, batch_size = 16,
                            validation_split = 0.2, verbose =2,
                            callbacks = [es, cp, ts_board])

#4. evaluate
loss_acc = model.evaluate(x_test, y_test, batch_size =16)
print('loss_acc :', loss_acc)

y_pred = model.predict(x_test)

# RMSE
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: ', rmse)

# r2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print('R2: ', r2)


# graph
import matplotlib.pyplot as plt
plt.figure(figsize = (10,6))

# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.',c= 'blue', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['mse'], marker = '.', c = 'red', label = 'mse')
plt.plot(hist.history['val_mse'], marker = '.',c= 'blue', label = 'val_mse')
plt.grid()
plt.title('mse')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.legend()

plt.show()

# R2:  0.744981538124726