from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC, LinearSVC 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1. data
cancer = load_breast_cancer()

x = cancer.data
y = cancer.target 

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1, train_size =0.8)

#2. model
# model =SVC()                                     
# model = LinearSVC()                                       
# model = KNeighborsClassifier(n_neighbors = 1)
# model = RandomForestClassifier()
# model = KNeighborsRegressor(n_neighbors = 1)    # 회귀모델  
model = RandomForestRegressor()                   # ValueError : accuracy_score를 지우면 사용할 수 있다.

#3. fit
model.fit(x_train, y_train)

#4. predict
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print('acc: ', acc)
# acc:  0.9736842105263158
# acc:  0.9736842105263158
# acc:  0.9298245614035088
# acc:  0.956140350877193
# acc:  0.9298245614035088              # acc값
# ValueError: Classification metrics can't handle a mix of binary and continuous targets

score = model.score(x_test, y_test)
print('score: ', score)
# score:  0.9736842105263158
# score:  0.9736842105263158
# score:  0.9298245614035088
# score:  0.956140350877193 
# score:  0.6984126984126984             # r2값        
# ValueError: Classification metrics can't handle a mix of binary and continuous targets