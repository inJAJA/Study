from sklearn.datasets import load_iris
from sklearn.svm import SVC, LinearSVC 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1. data
iris = load_iris()

x = iris.data
y = iris.target

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1, train_size =0.8)

print(x.shape)
print(y.shape)

#2. model
# model =SVC()                                     # 원 핫 인코딩 필요 없음
# model = LinearSVC()                                       
# model = KNeighborsClassifier(n_neighbors = 1)
# model = RandomForestClassifier() 
# model = KNeighborsRegressor(n_neighbors = 1)                
model = RandomForestRegressor()                    # ValueError : accuracy_score를 지우면 사용할 수 있다.
        # 분류를 회귀 모델로 돌릴 수 는 있지만 정확한 값이 

#3. fit
model.fit(x_train, y_train)

#4. predict
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
# print('acc: ', acc)
# acc:  0.8
# acc:  0.9666666666666667
# acc:  1.0
# acc:  0.9666666666666667
# acc:  1.0
# ValueError: Classification metrics can't handle a mix of multiclass and continuous targets
  # gh

score = model.score(x_test, y_test)                # 회귀 모델이면 R2 값
print('score: ', score)                            # 분류 모델이면 ACC 값 반환
# score:  0.8
# score:  0.9666666666666667
# score:  1.0                                      # 운이 좋게 acc와 r2값이 동일하게 나온 것
# score:  0.9666666666666667
# score:  1.0
# ValueError: Classification metrics can't handle a mix of multiclass and continuous targets