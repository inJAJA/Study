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

# x_val, x_test, y_val, y_test = train_test_split( 
#     # x_test, y_test, random_state=66,
#     x_test, y_test, shuffle = False,
#     test_size =0.5
#     )    

# x_train = x[:60]  # python시퀀스 자료형 슬라이스 참조
# x_val = x[60:80]
# x_test = x[80:]

# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]

print(x_train)
print(x_test )


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(5, input_dim = 3 ))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  # metrics가 돌아가는데 딜레이가 있음 
model.fit(x_train, y_train, epochs =1, batch_size =1,
        # validation_data = (x_val, y_val)
          validation_split= 0.25, verbose=3  #  0 : 안보임                      
          )                                  #  1 : defalut             
                                             #  2 : 프로그래스 바가 안보임
                                             #  3 : epoch만 보임 

#4. 평가,예측
loss, mse = model.evaluate(x_test, y_test, batch_size =1) 
print("loss : ", loss)
print("mse : ", mse)

# y_pred = model.predict(x_pred)  #눈으로 보기 위한 예측값
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

