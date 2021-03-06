import numpy as np
from keras.models import Sequential, Model
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


# reshape( , , )
x = x.reshape(x.shape[0], x.shape[1], 1)
# x = np.reshape(x, (6, 4, 1))
# x = x.reshape(6, 4, 1)


print(x.shape)
print(y.shape)


#==================================================================================================
#2. 모델
""" 전위 학습 : 저장한 model 불러오기 """ 
from keras.models import load_model                 # load_model 가져오기
model = load_model('./model/save_keras44.h5')       # load_model( '경로 / 저장한 파일 이름' ) : model 불러오기 
                            
                
'''남의 모델을 가져와서 사용해도 내가 가지고 있는 데이터가 다르기 때문에 튜닝해 줘야 한다.'''
model.add(Dense(40, name ='new1'))                  # 가져온 모델의 layer에서 이미 쓴 이름이 나오기 때문에 (중복)
model.add(Dense(30, name ='new2'))                  # 이름을 다르게 지정해 줘야 한다.
model.add(Dense(20, name ='new3'))
model.add(Dense(10, name ='new4'))
model.add(Dense(1, name ='new5'))

model.summary()


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


y_predict = model.predict(x)
print(y_predict)
