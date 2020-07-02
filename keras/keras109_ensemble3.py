#1. 데이터
import numpy as np
x1_train = np.array([1,2,3,4,5,6,7,8,9,10])
x2_train = np.array([1,2,3,4,5,6,7,8,9,10])

y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])

#2.모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate

input1 = Input(shape=(1,))
x1 = Dense(100)(input1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)

input2 = Input(shape=(1,))
x2 = Dense(100)(input2)
x2 = Dense(100)(x2)
x2 = Dense(100)(x2)

merge = concatenate([x1, x2])

x3 = Dense(50)(merge)
output1 = Dense(1)(x3)

x4 = Dense(70)(merge)
x4 = Dense(70)(x4)
output2 = Dense(1, activation='sigmoid')(x4)

model = Model(inputs = [input1, input2], outputs= [output1, output2])
model.summary()

#3.훈련
model.compile(loss = ['mse', 'binary_crossentropy'], 
              optimizer='adam',
              loss_weights = [0.1, 0.9], # output2에 loss 가중치를 0.9 주겠다. 
              metrics=['mse', 'acc'])

model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=1)

#4,평가, 예측

loss = model.evaluate([x1_train, x2_train],[y1_train,y2_train])
print("loss: ", loss)       
# 전체 loss, output1의 loss, output2의 loss, output1의 mse, output1의 acc, output2의 mse, output2의 acc                     
# 전체 loss = output1의 loss * 0.1          + output2의 loss * 0.9
#                             loss_weight                     loss_weight
# loss_weight를 안 쓴것보다 치우침이 덜 하지만 그래도 쏠린 비중이 크다.
# 모델 두개를 돌리는 것이 더 좋다.

x1_pred = np.array([11,12,13,14])
x2_pred = np.array([11,12,13,14])

y_pred = model.predict([x1_pred, x2_pred])
print(y_pred)