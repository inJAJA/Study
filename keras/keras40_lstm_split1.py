
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))
size = 5                                         


# LSTM 모델을 완성하시오.
"""
def split_x(seq, size):
    xxx=[]
    for i in range(len(seq) - size + 1):       # len = length  : 길이  i in range(6)  : [0, 1, 2, 3, 4, 5]
        subset = seq[i : (i + size -1)]    
                                               # i =0,  subset = a[ 0 : 5 ] = [ 1, 2, 3, 4, 5]
        xxx.append([item for item in subset])  # aaa = [[1, 2, 3, 4, 5]]
        #aaa.append([subset])          
    print(type(xxx))
    return np.array(xxx)

def split_y(seq, size):
    yyy = []
    for i in range(len(seq) - size + 1):      
        y = seq[(i + size-1)]
        print(y)             
        yyy.append(y) 
    print(type(yyy))
    return np.array(yyy)

x = split_x(a, size)                       
print(x.shape)                                # (6, 4)   # time_steps = 4
y = split_y(a, size)                       
print(y.shape)                                # (6, )


"""
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):       # len = length  : 길이  i in range(6)  : [0, 1, 2, 3, 4, 5]
        subset = seq[i : (i + size)]           # i =0,  subset = a[ 0 : 5 ] = [ 1, 2, 3, 4, 5]
        aaa.append([item for item in subset])  # aaa = [[1, 2, 3, 4, 5]]
        # aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset)                                 
print(dataset.shape)                           # (6, 5)
print(type(dataset))                           # numpy.ndarray


# x, y 값 나누기
x = dataset[:, 0:4]                            # [ : ] 모든행 가져오고, [0 : 4] 0~3까지
y = dataset[:, 4]                              # [ : ] 모든행 가져오고, [  : 4] 4번째


# reshape( , , )
x = x.reshape(x.shape[0], x.shape[1], 1)
# x = np.reshape(x, (6, 4, 1))
# x = x.reshape(6, 4, 1)

x_predict = x_predict.reshape(1, 4, 1)   # x값 (6, 4, 1)와 동일한 shape로 만들어 주기 위함
                                         # (1, 4, 1) : 확인 1 * 4 * 1 = 4
# x_predict = x_predict.reshape(1, x_predict.shape[0], 1)

print(x.shape)
print(x_predict.shape)
print(y.shape)


#==================================================================================================

#2. 모델

model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape = (4, 1)))
model.add(LSTM(400, input_length =4, input_dim= 1))                # input_length : time_step (열)
model.add(Dense(300))   
model.add(Dense(200))   
model.add(Dense(100))   
model.add(Dense(100))   
model.add(Dense(50))   
model.add(Dense(1))

# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=100, mode = 'min')


#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics= ['mse'])
model.fit(x, y, epochs =3000, batch_size = 32,
         callbacks = [es])                

""" shape의 batch_size는 총 데이터 갯수
      fit의 batch_size는 한번에 가져와 훈현시키는 양
"""

#4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size= 32)

print('loss :',loss )
print('mse :',mse )


y_predict = model.predict(x)
print(y_predict)
