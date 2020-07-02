#1. 데이터
import numpy as np
x = np.array([1, 2, 3, 4])
y = np.array([1, 2, 3, 4])

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim = 1, activation = 'relu'))
model.add(Dense(3))
model.add(Dense(11))
model.add(Dense(1))

from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam # lr = learning_rate : 경사하강법
# optimizer = Adam(lr = 0.001)           # 0.010150063782930374
# optimizer = RMSprop(lr = 0.001)        # 0.0696386992931366
# optimizer = SGD(lr = 0.001)            # 0.011646460741758347
# optimizer = Adadelta(lr = 0.001)       # 8.886947631835938
# optimizer = Adagrad(lr = 0.001)        # 5.722466468811035
# optimizer = Nadam(lr = 0.001)          # 0.00925517175346613, 0.08171126246452332

optimize = [Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam]
for optimizer in optimize:
    optimizer = optimizer(lr = 0.001)
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse'])

    model.fit(x, y, epochs = 100, verbose = 0)

    loss = model.evaluate(x, y)
    print('loss :', loss)

    pred1 = model.predict([3.5])
    print('pred1 :',pred1)
    print('---------------------------')

# 4/4 [==============================] - 0s 3ms/step
# loss : [0.007180371321737766, 0.007180371321737766]
# pred1 : [[3.4622052]]
# ---------------------------
# 4/4 [==============================] - 0s 3ms/step
# loss : [0.001533465227112174, 0.001533465227112174]
# pred1 : [[3.450668]]
# ---------------------------
# 4/4 [==============================] - 0s 3ms/step
# loss : [4.5175401197639076e-08, 4.5175401197639076e-08]
# pred1 : [[3.5000975]]
# ---------------------------
# 4/4 [==============================] - 0s 3ms/step
# loss : [2.705360557797576e-08, 2.705360557797576e-08]
# pred1 : [[3.5000775]]
# ---------------------------
# 4/4 [==============================] - 0s 3ms/step
# loss : [1.1725816762009345e-07, 1.1725816762009345e-07]
# pred1 : [[3.4998322]]
# ---------------------------
# 4/4 [==============================] - 0s 3ms/step
# loss : [0.0004696653049904853, 0.0004696653049904853]
# pred1 : [[3.472844]]
# ---------------------------