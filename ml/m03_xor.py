from sklearn.svm import LinearSVC              
from sklearn.metrics import accuracy_score

#1. data
x_data = [[0, 0],[1, 0],[0, 1],[1,1]]         
y_data = [0, 1, 1, 0]                            # xor 연산

#2. model 
model = LinearSVC()                              # 사용 모델 명시

#3. fit
model.fit(x_data, y_data)

#4. evaluate, predict
x_test = [[0, 0], [1, 0], [0, 1],[1, 1]]
y_predict = model.predict(x_test)

                     
acc = accuracy_score([0, 1, 1, 0], y_predict)     # evaluate = score()
                     #  y_data

print(x_test, '의 예측 결과: ', y_predict)
print('add = ', acc)
# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과:  [0 0 0 0]    
# add =  0.5