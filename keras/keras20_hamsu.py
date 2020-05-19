#1. 데이터
import numpy as np
x = np.transpose([range(1, 101), range(311, 411), range(100)])  
y = np.transpose(range(711, 811))

print(x.shape)   

from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(  
    # x, y, random_state=66, shuffle = True,
    x, y, shuffle = False,
    train_size =0.8                                     
    )


#2. 모델구성
from keras.models import Sequential, Model      # 함수형 모델 가져오기 : Model
from keras.layers import Dense, Input           # 함수형 모델은 input layer 명시해야함
# model = Sequential()
# model.add(Dense(5, input_dim = 3 ))
# model.add(Dense(4))
# model.add(Dense(1))

# 함수형 모델은 각 layer의 이름을 명시해야함
input1 = Input(shape =(3, ))                     # input layer : 함수형 모델에서는 shape 사용, 행을 뺀 나머지 부분 

dense1 = Dense(10, activation = 'relu')(input1)  # 출력값 5, 함수형은 input이 무엇인지 명시해야함 :input1
dense1 = Dense(20, activation = 'relu')(dense1)  # acivation deflaut = linear
dense1 = Dense(30, activation = 'relu')(dense1)
dense1 = Dense(40, activation = 'relu')(dense1)
dense1 = Dense(30, activation = 'relu')(dense1)
dense1 = Dense(20, activation = 'relu')(dense1)
dense1 = Dense(10, activation = 'relu')(dense1)
dense1 = Dense(10, activation = 'relu')(dense1)

output1 = Dense(1)(dense1)                       # output layer

model = Model(inputs = input1, outputs= output1) # 함수형 모델이라고 정의 / 시퀀스 모델의 경우 ex) model = Sequential()
                                                 # inputs = input layer이름
                                                 # outputs = output layer 이름

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit(x_train, y_train, epochs =200, batch_size =1,
        # validation_data = (x_val, y_val)
          validation_split= 0.25, verbose=1 
)

#4. 평가,예측
loss, mse = model.evaluate(x_test, y_test, batch_size =1) 
print("loss : ", loss)
print("mse : ", mse)

# y_pred = model.predict(x_pred)                 #눈으로 보기 위한 예측값
# print("y_pred : ", y_pred)

y_predict = model.predict(x_test)  
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


