from sklearn.svm import LinearSVC               # SVM(Support vector machine)
                                                # : 결정 경계(desicion boundart) 분류를 위한 기준 선을 정의하는 모델
from sklearn.metrics import accuracy_score

#1. data
x_data = [[0, 0],[1, 0],[0, 1],[1,1]]         
y_data = [0, 0, 0, 1]                            # and 연산

#2. model 
model = LinearSVC()                              # 사용 모델 명시

#3. fit
model.fit(x_data, y_data)

#4. evaluate, predict
x_test = [[0, 0], [1, 0], [0, 1],[1, 1]]
y_predict = model.predict(x_test)

                     
acc = accuracy_score([0, 0, 0, 1], y_predict)     # evaluate = score()
                     #  y_data

print(x_test, '의 예측 결과: ', y_predict)
print('add = ', acc)