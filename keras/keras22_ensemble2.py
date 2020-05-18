'''
<앙상블> 
: 각자 모델을 훈련시키고 합치는 것 
'''
#1. 데이터
import numpy as np
x1 = np.transpose([range(1, 101), range(311, 411)])  
x2 = np.transpose([range(711, 811), range(711,811)])

y1 = np.transpose([range(101, 201), range(411,511)])
y2 = np.transpose([range(501, 601), range(711,811)])       
y3 = np.transpose([range(411, 511), range(611,711)])       

################## 여기서 부터 수정 #################
from sklearn.model_selection import train_test_split    
x1_train, x1_test, y1_train, y1_test = train_test_split(  
    # x, y, random_state=66, shuffle = True,
    x1, y1, shuffle = False,
    train_size =0.8                                     
    )
   
x2_train, x2_test, y2_train, y2_test = train_test_split(  
    # x, y, random_state=66, shuffle = True,
    x2, y2, shuffle = False,
    train_size =0.8                                     
    )    

y3_train, y3_test = train_test_split(
    y3, shuffle = False,
    train_size = 0.8
)


#2. 모델구성
from keras.models import Sequential, Model 
from keras.layers import Dense, Input 

######### 모델 1 #########
input1 = Input(shape =(2, ))           
dense1_1 = Dense(7, activation = 'relu')(input1)
dense1_2 = Dense(10, activation = 'relu' )(dense1_1)
dense1_2 = Dense(5, activation = 'relu')(dense1_1)

   

######### 모델 2 #########
input2 = Input(shape =(2, )) 
dense2_1 = Dense(8, activation = 'relu')(input2) 
dense2_2 = Dense(12, activation = 'relu')(dense2_1)
dense2_2 = Dense(4, activation = 'relu')(dense2_1)
  

######### 모델 병합#########
from keras.layers.merge import concatenate   
merge1 = concatenate([dense1_2, dense2_2])   # list형태로 묶임 

middle1 = Dense(10)(merge1)
middle1 = Dense(10)(middle1) 
middle1 = Dense(7)(middle1)

######### output 모델 구성 ###########

output1 = Dense(30)(middle1)   
output1_2 = Dense(15)(output1)
output1_3 = Dense(2)(output1_2)  

output2 = Dense(25)(middle1)   
output2_2 = Dense(12)(output2)
output2_3 = Dense(2)(output2_2) 

output3 = Dense(20)(middle1)
output3_2 = Dense(9)(output3)
output3_3 = Dense(2)(output3_2)

######### 모델 명시 #########
model = Model(inputs = [input1, input2],
              outputs= [output1_3, output2_3, output3_3]) 

model.summary() 


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit([x1_train, x2_train], 
          [y1_train, y2_train, y3_train], epochs =100, batch_size =1,
        # validation_data = (x_val, y_val)
        # validation_split= 0.25, verbose=1
)

#4. 평가,예측
loss = model.evaluate([x1_test, x2_test],
                      [y1_test, y2_test, y3_test], batch_size =1)

# print("model.metrics_names : ", model.metrics_names) 

print("loss : ", loss)                               
# print("mse : ", mse)

# y_pred = model.predict(x_pred)  #눈으로 보기 위한 예측값
# print("y_pred : ", y_pred)

y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])  
print(y1_predict)
print(y2_predict)
print(y3_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE3 = RMSE(y3_test, y3_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE3 : ", RMSE3)
print("RMSE : ", (RMSE1 + RMSE2 + RMSE3)/3 )

# R2 구하기
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
r2_3 = r2_score(y3_test, y3_predict)
print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2_3 : ", r2_3)
print("R2 : ", (r2_1 + r2_2 + r2_3)/3)




