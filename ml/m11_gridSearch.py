import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score   # Kfold : 교차 검증
from sklearn.model_selection import GridSearchCV   # CV = cross validation

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# gridSearch
# 내가 정해놓은 조건들을 충족하는 것을 전부다 가져온다. 


#1. data
iris = pd.read_csv('D:/Study/data/csv/iris.csv', header = 0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 44)  # train, test

parameters = [ 
    {"C": [1, 10, 100, 1000], "kernel" : ["linear"]},                           
    {"C": [1, 10, 100, 1000], "kernel" : ["rbf"], "gamma":[0.001, 0.0001]},      
    {"C": [1, 10, 100, 1000], "kernel" : ["sigmoid"], "gamma":[0.001, 0.0001]}    
]       # 가중치 인수         # 커널 트릭              # = running rate

kfold = KFold(n_splits = 5, shuffle = True)                                                     # train, validation
    
        #  진짜 모델,     그 모델의 파라미터 , cross_validtion 수
model = GridSearchCV(SVC(), parameters, cv =  kfold)  # SVC()모델을 가지고 parameters를 조정하고, kfold만큼 
     


model.fit(x_train, y_train)

print('최적의 매개변수 : ', model.best_estimator_)
y_pred = model.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred) )

