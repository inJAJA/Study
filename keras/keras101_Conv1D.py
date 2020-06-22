
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten

#1. 데이터
a = np.array(range(1,101))
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
print(dataset.shape)                           # (96, 5)
print(type(dataset))                           # numpy.ndarray


"""실습 1. train, test 분리할 것.                 (90행) 8 : 2 비율
   실습 2. 마지막 6개의 행을 predict로 만들고 싶다.
   실습 3. validatoion 을 넣을 것                 (train의 20%)
"""

## x, y 값 나누기
x = dataset[:90, 0:4]                            # [ : ] 모든행 가져오고, [0 : 4] 0~3까지
y = dataset[:90, 4]                              # [ : ] 모든행 가져오고, [  : 4] 4번째
print(x.shape)                          
print(y.shape)

x_predict = dataset[-6:, 0:4]
# y_predict = dataset[-6:, 4]                    # model로 에측 할 것이기 때문에 필요 없다.
print(x_predict.shape)


# reshape( , , )
x = x.reshape(90, 4, 1)
x_predict = x_predict.reshape(6, 4, 1)

# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 66,  train_size = 0.8)


print(x_train.shape)


#==================================================================================================
#2. 모델

model = Sequential()
# model.add(LSTM(200, input_shape= (4, 1)))   
model.add(Conv1D(140, 2, padding = 'same', input_shape = (4,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(50, 2, padding = 'same'))
model.add(Flatten())
model.add(Dense(90))   
model.add(Dense(90))   
model.add(Dense(80))   
model.add(Dense(50))   
model.add(Dense(10))   
model.add(Dense(1))

model.summary()

# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=100, mode = 'min')


#3. 실행
model.compile(loss = 'mse', optimizer='adam', metrics= ['mse'])
model.fit(x_train, y_train, epochs =800, batch_size = 16 , validation_split= 0.2, 
         shuffle= True,                                  # fit에 shuffle 추가 가능, random_state는 사용 불가
         callbacks = [es])                 



#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size= 16)

print('loss :',loss )
print('mse :',mse )


y_predict = model.predict(x_predict)
print(y_predict)
'''
loss : 1.938651904412028e-05
mse : 1.9386519852560014e-05
[[ 95.00226 ]
 [ 96.002235]
 [ 97.0022  ]
 [ 98.00216 ]
 [ 99.00211 ]
 [100.002075]]
'''