'''
<앙상블> 
: 각자 모델을 훈련시키고 합치는 것 
'''
#1. 데이터
import numpy as np
x1 = np.transpose([range(1, 101), range(311, 411), range(100)])  
y1 = np.transpose([range(711, 811), range(711,811), range(100)])

x2 = np.transpose([range(101, 201), range(411,511), range(100,200)])
y2 = np.transpose([range(501, 601), range(711,811), range(100)])       # W와 bias 값이 같을 필요는 없다./ 데이터 개수만 같으면 됌


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


#2. 모델구성
from keras.models import Sequential, Model 
from keras.layers import Dense, Input 

######### 모델 1 #########
input1 = Input(shape =(3, ))           
dense1_1 = Dense(7, activation = 'relu', name = 'Dense1_1')(input1) # name = ' ' : layer 이름 바꾸기
dense1_2 = Dense(5, activation = 'relu', name = 'Dense1_2')(dense1_1)
   

######### 모델 2 #########
input2 = Input(shape =(3, )) 
dense2_1 = Dense(8, activation = 'relu')(input2) 
dense2_2 = Dense(4, activation = 'relu')(dense2_1)
  

######### 모델 병합#########
from keras.layers.merge import concatenate   # 레이어를 병합 해준다.
merge1 = concatenate([dense1_2, dense2_2])   # 각 모델의 마지막 layer를 input으로 넣어줌 : list형태
                                             # concatenate에서는 param연산이 이루어 지지 않는다. 

middle1 = Dense(30)(merge1)
middle1 = Dense(5)(middle1)      # 병합 후 새로운 layer 설정 가능
middle1 = Dense(7)(middle1)

######### output 모델 구성 ###########

output1 = Dense(30)(middle1)     # 상단 레이어를 input에 넣는다.
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)  # 모델 1의 마지막 output

output2 = Dense(25)(middle1)     # 상단 레이어를 input를 넣는다.
output2_2 = Dense(5)(output2)
output2_3 = Dense(3)(output2_2)  # 모델 2의 마지막 output

######### 모델 명시 #########
model = Model(inputs = [input1, input2],
              outputs= [output1_3, output2_3]) 

model.summary() # 두 모델의 layer가 번갈아 나온다. 


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit([x1_train, x2_train], 
          [y1_train, y2_train], epochs =1, batch_size =1,
        # validation_data = (x_val, y_val)
        # validation_split= 0.25, verbose=1
)

#4. 평가,예측
loss = model.evaluate([x1_test, x2_test],
                      [y1_test, y2_test], batch_size =1)

print("model.metrics_names : ", model.metrics_names) # model.metrics_names : evaluate값으로 무엇이 나오는지 알려줌
                                                     # [(총loss값), (loss 1), (loss 2), (metrics 1), (metrics 2)]

# loss : 모델이 다중 아웃풋을 갖는 경우, 손실의 리스트 혹은 손실의 딕셔너리를 전달하여 각 아웃풋에 각기 다른 손실을 사용할 수 있습니다.
#       따라서 모델에 의해 최소화되는 손실 값은 모든 개별적 손실의 합이 됩니다.

print("loss : ", loss)                               
# print("mse : ", mse)

# y_pred = model.predict(x_pred)  #눈으로 보기 위한 예측값
# print("y_pred : ", y_pred)

y1_predict, y2_predict = model.predict([x1_test, x2_test])  
print(y1_predict)
print(y2_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE : ", (RMSE1 + RMSE2)/2 )

# R2 구하기
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2_2 : ", (r2_1 + r2_2)/2)




