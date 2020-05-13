from keras.models import Sequential  # keras안에서 .(가져오기) model 중 import(가장 끝에거 가져오기) Sequential 
from keras.layers import Dense       # keras안에서 layer중  Dense형을 사용하겠다.
import numpy as np                   # numpy를 가져와 사용하겠다. numpy를 np라고 줄여 쓰겠다.

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

model = Sequential() # Seqeuntial을 model이라 하겠다.
model.add(Dense(5, input_dim =1, activation='relu')) # model(Seqeuntial)에서 add사용, input_dimension=1 1차원
model.add(Dense(3))                                  # Sequntial이라 input=5, output=3
model.add(Dense(1, activation='relu'))                

model.summary()   
#모델의 노드와 파라미터의 수 등을 확인
# paramter : node와 node가 연결된 선의 갯수 / = [input node * output node] + bias(1*output node)  
#                                       or (input node + 1(bias)) * output node


model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])  
#             손실 = mse, 손실최적화 = adam, metrics을 accuracy로 보여주기

model.fit(x_train, y_train, epochs =100, batch_size=1, validation_data = (x_train, y_train))
# 훈련시키기(x_train, y_train으로 훈련, 훈련횟수, )
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

print("loss : ", loss)
print("acc : ", acc)

output = model.predict(x_test)
print("결과물 : |n", output)

