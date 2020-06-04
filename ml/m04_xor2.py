from sklearn.svm import LinearSVC                  # 선형분류에 특화
from sklearn.svm import SVC                        # 다항식 커널
from sklearn.neighbors import KNeighborsClassifier # 군집 분석

from sklearn.metrics import accuracy_score


#1. data
x_data = [[0, 0],[1, 0],[0, 1],[1,1]]         
y_data = [0, 1, 1, 0]                               # xor 연산

#2. model 
# model = LinearSVC()                               # 사용 모델 명시
# model = KNeighborsClassifier(n_neighbors =1)
model = SVC()


#3. fit
model.fit(x_data, y_data)

#4. evaluate, predict
x_test = [[0, 0], [1, 0], [0, 1],[1, 1]]
y_predict = model.predict(x_test)

                     
acc = accuracy_score([0, 1, 1, 0], y_predict)        # evaluate = score()
                     #  y_data

print(x_test, '의 예측 결과: ', y_predict)
print('add = ', acc)
# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과:  [0 1 1 0]    
# add =  1.0