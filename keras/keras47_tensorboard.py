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
print(dataset.shape)                           # (6, 5)


# x, y 값 나누기
x = dataset[:, 0:4]                            # [ : ] 모든행 가져오고, [0 : 4] 0~3까지
y = dataset[:, 4]                              # [ : ] 모든행 가져오고, [  : 4] 4번째


# reshape( , , )
x = x.reshape(x.shape[0], x.shape[1], 1)


x_predict = x_predict.reshape(1, 4, 1)   # x값 (6, 4, 1)와 동일한 shape로 만들어 주기 위함
                                         # (1, 4, 1) : 확인 1 * 4 * 1 = 4
# x_predict = x_predict.reshape(1, x_predict.shape[0], 1)


#==================================================================================================
#2. 모델
""" 저장한 model 불러오기 """
# from keras.models import load_model
# model = load_model('./model/save_keras44.h5')
model = Sequential()
model.add(LSTM(5, input_shape =(4,1)))                # input_length : time_step (열)
model.add(Dense(3))   
model.add(Dense(1, name ='new'))                 

model.summary()


# EarlyStopping
from keras.callbacks import EarlyStopping        
es = EarlyStopping(monitor = 'loss', patience=100, mode = 'min')


""" Tensorboard """
from keras.callbacks import TensorBoard                                 # Tensorboard 가져오기
tb_hist = TensorBoard(log_dir='graph', histogram_freq= 0 ,              # log_dir=' 폴더 ' : 제일 많이 틀림
                      write_graph= True, write_images= True)           
"""
: 웹 상에서 graph를 볼 수 있게 만들어줌( log_dir=' 폴더 '에 tensorboard의 로그 저장 )
: loss와 metrics 값 보여줌
 # cmd 창에서 
 1. d:
 2. cd Study                                # 경로가 제일 많이 틀린다.
 3. cd graph
 4. tnsorboard --logdir=.                   # 내 컴퓨터에서 tensorboard를 쓰겠다.
 =>  127.0.0.1:6006 를 웹주소에 친다.
     127.0.0.1 : 내 컴퓨터 IP
     6006      : 포트 번호    -> 내 pc의 6006번 포트를 사용하겠다.
"""


# 가중치(W) 저장 방법 
#3. 실행
model.compile(loss = 'mse', optimizer='adam', metrics= ['acc'])
hist = model.fit(x, y, epochs =100, batch_size = 32, verbose =1,   
                 validation_split = 0.2,
                 callbacks = [es, tb_hist])                              # Tensorboard 사용
# hist = model.fit에 훈련시키고 난 loss, metrics안에 있는 값들을 반환한다.


print(hist)   #자료형 모양 <keras.callbacks.callbacks.History object at 0x00000178F0734EC8> : 원래 안보여줌
print(hist.history.keys())                          # dict_keys(['loss', 'mse'])
 

# 그래프를 그려서 보가
import matplotlib.pyplot as plt                     # 그래프 그리는 것

plt.plot(hist.history['loss'])                      # 'loss'값을 y로 넣겠다./ 인자 하나만 쓰면 y 값으로 들어감
plt.plot(hist.history['val_loss'])                  # 시간에 따른 loss, acc여서 x 값으로는 자연스럽게 epoch가 들어감
plt.plot(hist.history['acc']) 
plt.plot(hist.history['val_acc']) 
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss','val loss','train acc','val acc'])    # 선에 대한 색깔과 설명이 나옴
# plt.show()                                                     # 그래프 보여주기

'''
#4. 평가, 예측
loss, mse = model.evaluate(x, y)

print('loss :',loss )
print('mse :',mse )


y_predict = model.predict(x_predict)
print(y_predict)
'''