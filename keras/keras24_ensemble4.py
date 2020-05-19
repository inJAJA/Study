'''
<앙상블> 
: 각자 모델을 훈련시키고 합치는 것 
'''
#1. 데이터
import numpy as np
x1 = np.transpose([range(1, 101), range(301,401)])  # (100, 2)

y1 = np.transpose([range(711, 811), range(611,711)])
y2 = np.transpose([range(101, 201), range(411,511)])


from sklearn.model_selection import train_test_split    
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(  
    # x, y, random_state=66, shuffle = True,
    x1, y1, y2, random_state = 5,
    train_size =0.8                                     
    )
   
# x2_train, x2_test = train_test_split(  
#     # x, y, random_state=66, shuffle = True,
#     x2, shuffle = False,
#     train_size =0.8                                     
#     )    

print(x1_train.shape)   # (80, 2)
print(y1_test.shape)    # (20, 2)


#2. 모델구성
from keras.models import Model 
from keras.layers import Dense, Input 

######### 모델 1 #########
input1 = Input(shape =(2, ))                       # input layer
dense1_1 = Dense(10, activation = 'relu')(input1)
dense1_1 = Dense(100, activation = 'relu')(dense1_1)
dense1_1 = Dense(200, activation = 'relu')(dense1_1)
dense1_1 = Dense(200, activation = 'relu')(dense1_1)
dense1_1 = Dense(150, activation = 'relu')(dense1_1)
dense1_1 = Dense(100, activation = 'relu')(dense1_1)
dense1_2 = Dense(10, activation = 'relu')(dense1_1)
   
######### 모델 병합#########
# from keras.layers.merge import concatenate   
# merge1 = concatenate(dense1_2)   

# middle1 = Dense(13)(merge1)
# middle1 = Dense(11)(middle1) 
# middle1 = Dense(7)(middle1) 

######### output 모델 구성 ###########

output1 = Dense(10)(dense1_2)     # input 1 : dense1_2 에서 분기
output1_1 = Dense(10)(output1)   
output1_1 = Dense(10)(output1_1) 
output1_1 = Dense(10)(output1_1)   
output1_2 = Dense(5)(output1_1)
output1_3 = Dense(2)(output1_2)   # output 1  / output layer

output2 = Dense(10)(dense1_2)     # input 2 : dens1_2 에서 분기
output2_1 = Dense(10)(output2)  
output2_1 = Dense(10)(output2_1) 
output2_1 = Dense(10)(output2_1)   
output2_2 = Dense(5)(output2_1)
output2_3 = Dense(2)(output2_2)   # output 2 / output layer


######### 모델 명시 #########
model = Model(inputs = input1,
              outputs= [output1_3, output2_3]) 

model.summary()                    # shape 조심 : input, output 잘 보기

# # Early Stopping
# from keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 50, verbose =1)


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit(x1_train,
        [y1_train, y2_train], epochs= 280, batch_size =1,
        # validation_data = (x_val, y_val)
        validation_split= 0.25, verbose=1 #,callbacks = [es]
        )




#4. 평가,예측
loss = model.evaluate(x1_test,
                     [y1_test, y2_test], batch_size =1)

# print("model.metrics_names : ", model.metrics_names) 

print("loss : ", loss)                               
# print("mse : ", mse)

# y_pred = model.predict(x_pred)  #눈으로 보기 위한 예측값
# print("y_pred : ", y_pred)

y1_predict, y2_predict = model.predict(x1_test)  
print(y1_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE : ", (RMSE1 + RMSE2)/2)


# R2 구하기
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2 : ", (r2_1 + r2_2)/2 )



