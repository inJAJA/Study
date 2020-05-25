import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))
size = 5                                         

x_predict = np.array([11, 12, 13, 14])

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
""" 저장한 model 불러오기 """
input = Dense(10)
from keras.models import load_model
model = load_model('./model/save_keras44.h5')


model.add(Dense(1, name ='new'))                    # 가져온 모델의 layer에서 이미 쓴 이름이 나오기 때문에 (중복)
                                                    # 이름을 다르게 지정해 줘야 한다.

model.summary()

'''
# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=100, mode = 'min')


#3. 실행
model.compile(loss = 'mse', optimizer='adam', metrics= ['mse'])
model.fit(x, y, epochs =3000, batch_size = 32,
         callbacks = [es])                


#4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size= 32)

print('loss :',loss )
print('mse :',mse )


y_predict = model.predict(x_predict)
print(y_predict)
'''