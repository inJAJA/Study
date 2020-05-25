
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM


#==================================================================================================
#2. 모델

model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape = (4, 1)))
model.add(LSTM(100, input_length =4, input_dim= 1))                # input_length : time_step (열)
model.add(Dense(50))     
model.add(Dense(30))   
model.add(Dense(20))     
model.add(Dense(10))

model.summary()

""" 모델 SAVE 방법 """
model.save(".//model//save_keras44.h5")     #  . : 현재 폴더,  
# model.save("./model/save_keras44.h5")     # // , / , \ : 하단 폴더
# model.save(".\model\save_keras44.h5")    

# model.save("경로 / 파일 이름 .h5")


print("저장 잘됬다")