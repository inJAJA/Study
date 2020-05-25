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


print(x.shape)
print(y.shape)


#==================================================================================================
#2. 모델
""" 저장한 model 불러오기 """
input = Dense(10)
from keras.models import load_model
model = load_model('./model/save_keras44.h5')


model.add(Dense(50, name ='new1'))                  # 가져온 모델의 layer에서 이미 쓴 이름이 나오기 때문에 (중복)
model.add(Dense(90, name ='new2'))                  # 이름을 다르게 지정해 줘야 한다.
model.add(Dense(80, name ='new3'))
model.add(Dense(50, name ='new4'))
model.add(Dense(30, name ='new5'))
model.add(Dense(1, name ='new6'))



model.summary()


# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=100, mode = 'min')



""" 가중치(W) 저장 방법 """
#3. 실행
model.compile(loss = 'mse', optimizer='adam', metrics= ['acc'])
hist = model.fit(x, y, epochs =2000, batch_size = 8, verbose =1,   
                 validation_split = 0.2,
                 callbacks = [es])                
# hist = model.fit에 훈련시키고 난 loss, metrics안에 있는 값들을 반환한다.


print(hist)   #자료형 모양 <keras.callbacks.callbacks.History object at 0x00000178F0734EC8> : 원래 안보여줌
print(hist.history.keys())                          # dict_keys(['loss', 'mse'])
 

# 그래프를 그려서 보가
import matplotlib.pyplot as plt                     # 그래프 그리는 것

plt.plot(hist.history['loss'])                      # 'loss'값을 y로 넣겠다./ 하나만 쓰면 y 값으로 들어감
plt.plot(hist.history['val_loss'])                  # 시간에 따른 loss, acc여서 x 값으로는 자연스럽게 epoch가 들어감
plt.plot(hist.history['acc']) 
plt.plot(hist.history['val_acc']) 
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss','val loss','train acc','val acc'])    # 선에 대한 색깔과 설명이 나옴
plt.show()                                          # 그래프 보여주기


#4. 평가, 예측
loss, mse = model.evaluate(x, y)

print('loss :',loss )
print('mse :',mse )


y_predict = model.predict(x)
print(y_predict)
