# 이진 분류
import numpy as np
#1. 데이터 
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

x = diabetes.data
y = diabetes.target

print(x.shape) # (442, 10)
print(y.shape) # (442,)

print(diabetes.feature_names)     
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

print(np.size(x, 1))          # 열의 개수 구하기

# scatter graph
import matplotlib.pyplot as plt
plt.figure(figsize = (20, 10))
for i in range(np.size(x, 1)):
    plt.subplot(2, 5, i+1)
    plt.scatter(x[:, i], y)
    plt.title(diabetes.feature_names[i])
plt.xlabel('columns')
plt.ylabel('target')
plt.axis('equal')
plt.legend()
plt.show()

# x
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state =100, 
                                                  train_size = 0.8)

#2. model
from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(100, input_shape= (10, ), activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(300, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(250, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(150, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(90, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(0, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation = 'relu'))


# callbacks
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
# earlystopping
es = EarlyStopping(monitor = 'val_loss', patience = 50, verbose = 1 )
# tensorboard
ts_board = TensorBoard(log_dir = 'graph', histogram_freq = 0,
                        write_graph = True, write_images = True)
# modelcheckpotin
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
ckpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                          save_best_only = True)


#3. complie, fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 64,
                validation_split = 0.2, verbose =2,
                callbacks = [es])

#4. evaluate, predict
loss, mse = model.evaluate(x_test, y_test, batch_size = 64) 
print('loss: ', loss)
print('mse: ', mse)

y_pred = model.predict(x_test)

# RMSE
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: ', rmse)

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print('R2: ', r2)

# graph
import matplotlib.pyplot as plt
# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], c= 'cyan', marker = 'o', label = 'loss')
plt.plot(hist.history['val_loss'], c= 'red', marker = 'o', label = 'val_loss')
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['mse'], c= 'cyan', marker = '+', label = 'mse')
plt.plot(hist.history['val_mse'], c= 'red', marker = '+', label = 'val_mse')
plt.title('mse')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.legend()
    