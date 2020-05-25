
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))
size = 5                                         

# LSTM 모델을 완성하시오.
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

'''
# reshape( , , )
x = x.reshape(x.shape[0], x.shape[1], 1)
# x = np.reshape(x, (6, 4, 1))
# x = x.reshape(6, 4, 1)
'''
print(x.shape)                          
print(y.shape)


#==================================================================================================
#2. 모델

model = Sequential()
model.add(Dense(100, input_dim= 4))                # input_length : time_step (열)
model.add(Dense(100))   
model.add(Dense(100))   
model.add(Dense(80))   
model.add(Dense(50))   
model.add(Dense(10))   
model.add(Dense(1))

# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=100, mode = 'min')


#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics= ['mse'])
model.fit(x, y, epochs =800, batch_size = 16 ,
         callbacks = [es])                

""" shape의 batch_size는 총 데이터 갯수
      fit의 batch_size는 한번에 가져와 훈현시키는 양
"""

#4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size= 16)

print('loss :',loss )
print('mse :',mse )


y_predict = model.predict(x)
print(y_predict)
