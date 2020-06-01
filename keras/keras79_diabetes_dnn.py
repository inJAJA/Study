
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

print(np.size(x, 1))          # 열의 개수 구하기 : 10

# scatter graph
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 7))
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
#'Sex' one hot encoding
sex_max = np.max(x[:, 1])   
sex_min = np.min(x[:, 1])
print(sex_max)                  #  0.0506801187398187
print(sex_min)                  # -0.044641636506989

for i in range(np.size(x, 0)):        # np.size(x, 0) : 의 행의 크기 
    if x[i, 1]==sex_min:
        x[i, 1]=0
    else:
        x[i, 1]=1

plt.figure(figsize = (10, 5))
plt.scatter(x[:, 1], y)
plt.title(diabetes.feature_names[1])
plt.xlabel('S4')
plt.ylabel('target')
plt.legend()
plt.show()

# # 'S4' 0, 1, 2, 3, 4, 5로 분류
# s4_max = np.max(x[:, -3])
# s4_min = np.min(x[:, -3])
# print(s4_max)
# print(s4_min)
# term = (0.1- s4_min)/10 

# for i in range(np.size(x, 0)):
#     if  x[i, -3] < (s4_min + term):
#         x[i, -3] = 0
#     elif (x[i, -3] >= (s4_min + term)) & (x[i, -3] < (s4_min + term*2)):
#         x[i, -3] = 1 
#     elif (x[i, -3] >= (s4_min + term*2)) & (x[i, -3] < (s4_min + term*3)):
#         x[i, -3] = 2
#     elif (x[i, -3] >= (s4_min + term*3)) & (x[i, -3] < (s4_min + term*4)):
#         x[i, -3] = 3 
#     elif (x[i, -3] >= (s4_min + term*4)) & (x[i, -3] < (s4_min + term*5)):
#         x[i, -3] = 4 
#     else:
#         x[i, -3] = 5

# plt.figure(figsize = (20, 10))
# plt.scatter(x[:, -3], y)
# plt.title(diabetes.feature_names[-3])
# plt.xlabel('S4')
# plt.ylabel('target')
# plt.legend()
# plt.show()


# scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state =100, 
                                                  train_size = 0.8)

# x1
x_train1 = x_train[:,:4]
x_test1 = x_test[:, :4]

# x2
x_train2 = x_train[:,4:]
x_test2 = x_test[:, 4:]


#2. model
from keras.models import Model
from keras.layers import Dense, Dropout, Input
# 1
input1 = Input(shape =(4, ))
dense1 = Dense(80, activation = 'relu')(input1)
desen1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
desen1 = Dropout(0.2)(dense1)

# 2 
input2 = Input(shape = (6, ))
dense2 = Dense(100, activation = 'relu')(input2)
desen2 = Dropout(0.2)(dense2)
dense2 = Dense(150, activation = 'relu')(dense2)
dense2 = Dropout(0.2)(dense2)
dense2 = Dense(100, activation = 'relu')(dense2)

# concentrate
from keras.layers.merge import concatenate   
merge1 = concatenate([dense1, dense2])
middle1 = Dense(300, activation='relu')(merge1)
middle1 = Dense(100, activation='relu')(middle1)
middle1 = Dense(50, activation='relu')(middle1)

# output
output1 = Dense(1, activation = 'relu')(middle1)

model = Model(inputs =[input1, input2], outputs = output1)


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
hist = model.fit([x_train1 ,x_train2] ,y_train, epochs = 100, batch_size = 64,
                validation_split = 0.2, verbose =2,
                callbacks = [es])

#4. evaluate, predict
loss, mse = model.evaluate([x_test1, x_test2], y_test, batch_size = 64) 
print('loss: ', loss)
print('mse: ', mse)

y_pred = model.predict([x_test1, x_test2])

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
    
plt.show()

"""
loss:  2634.910324953915
mse:  2634.910400390625
RMSE:  51.33137897972557
R2:  0.5041594238716269
"""
