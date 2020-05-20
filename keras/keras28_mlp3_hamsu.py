import numpy as np
x = np.transpose(range(1, 101))    
y = np.transpose([range(101, 201), range(711, 811), range(100)])

print(x.shape)  
print(y.shape)


from sklearn.model_selection import train_test_split     
x_train, x_test, y_train, y_test = train_test_split(  
    x, y, random_state=1, shuffle = True,
    # x, y, shuffle = False,
    train_size =0.8                                      
    )


# 모델
from keras.models import Model
from keras.layers import Dense, Input

input1 = Input(shape=(1,))
dense1 = Dense(4, activation = 'relu')(input1)
dense1 = Dense(10, activation = 'relu')(dense1)
dense1 = Dense(20, activation = 'relu')(dense1)
dense1 = Dense(22, activation = 'relu')(dense1) 
dense1 = Dense(30, activation = 'relu')(dense1)  
dense1 = Dense(17, activation = 'relu')(dense1)
dense1 = Dense(16, activation = 'relu')(dense1)
dense2 = Dense(10, activation = 'relu')(dense1)
dense3 = Dense(9, activation = 'relu')(dense2) 

output = Dense(3)(dense3)

model = Model(inputs = input1, outputs = output)

model.summary()

# 훈련
model.compile(loss = 'mse', optimizer = 'adam' ,metrics = ['mse'])

# Early Stopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 100, verbose = 1)

model.fit(x_train, y_train, epochs = 2000, batch_size = 2,
        validation_split = 0.25, callbacks = [es]
)

# 평가
loss = model.evaluate(x_test, y_test, batch_size =2)

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