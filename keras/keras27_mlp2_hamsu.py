import numpy as np
x = np.transpose([range(1, 101), range(311, 411), range(100)])  
y = np.transpose(range(711, 811))

print(x.shape)  # (100, 3) 
print(y.shape)  # (100, 1)

from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(  
    x, y, random_state=66, shuffle = True,
    # x, y, shuffle = False,
    train_size =0.8                                     
    )


# 모델
from keras.models import Model
from keras.layers import Dense, Input

input1 = Input(shape=(3,))                       # input layer
dense1 = Dense(10, activation = 'relu')(input1)
dense1 = Dense(10, activation = 'relu')(dense1)
dense1 = Dense(10, activation = 'relu')(dense1)
dense1 = Dense(10, activation = 'relu')(dense1)
dense1 = Dense(10, activation = 'relu')(dense1)
dense1 = Dense(10, activation = 'relu')(dense1)
dense2 = Dense(10, activation = 'relu')(dense1)
dense3 = Dense(9, activation = 'relu')(dense2)
output = Dense(1)(dense3)                        # output layer

model = Model(inputs = input1, outputs = output) # 함수형 model 명시

model.summary()

# 훈련
model.compile(loss = 'mse', optimizer = 'adam' ,metrics = ['mse'])

# Early Stopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 40, verbose = 1)

model.fit(x_train, y_train, epochs = 500, batch_size = 1,
        validation_split = 0.25, 
        callbacks = [es]
)

# 평가
loss = model.evaluate(x_test, y_test, batch_size =1)

y_predict = model.predict(x_test)
print(y_predict)


# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2 : ', r2)
