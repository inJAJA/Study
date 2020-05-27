# 이진 분류
"""train_test_predict분리하지 말고
딥러닝을 이용한 자연어 처리 입문 : 머신러닝 / 로지스틱 회귀 - 이진분류
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


# sigmoid 함수
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#1. 데이터
x = np.array(range(1, 11))
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

print(x.shape)                     # (10, )
print(y.shape)                     # (10, )


#2. 모델 구성

model = Sequential()
model.add(Dense(100,activation = 'relu', input_dim = 1))
model.add(Dense(90,activation = 'relu'))
model.add(Dense(70,activation = 'relu'))
model.add(Dense(60,activation = 'relu'))
model.add(Dense(50,activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(30,activation = 'relu'))
model.add(Dense(20,activation = 'relu'))
model.add(Dense(10,activation = 'relu'))     
model.add(Dense(1,activation = 'sigmoid'))
""" 
- 계산된 함수가 activation을 통해 다음 layer에 넘어간다.
- 가장 마지막 output layer값이 가중치와 '활성화 함수'와 곱해져서 반환된다. 
# sigmoid : 출력 값을 0과 1사이의 값으로 조정하여 반환한다.
"""


# EarlyStopping
# from keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor = 'loss', mode= 'auto', patience = 50, verbose =1)


#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])    # acc : 분류 모델 0 or 1
model.fit(x, y, epochs = 500, batch_size =32)
"""
# loss = 'binary_crossentropy'
: 이진 분류에서 이거만 써야함
: '이진 분류' 문제에 크로스 엔트로피 함수를 사용할 경우에 기재

# optimear = 'rmsprop'
"""

#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size= 32)
print('loss :', loss)
print('acc :', acc)


x_pred = np.array([1, 2, 3])
y_pred = model.predict(x_pred)
print('y_pred :', y_pred)
# sigmoid 함수를 거치지 않은 걸로 보여짐



y1_pred = np.where(y_pred >= 0.5, 1, 0)     
print('y_pred :', y1_pred)
"""
# np.where(조건, 조건에 맞을 때 값, 조건과 다를때 값)
: 조건에 맞는 값을 특정 다른 값으로 변환하기
"""