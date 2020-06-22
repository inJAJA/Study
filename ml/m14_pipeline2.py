# RandomiziedSearchCV
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC

#1. data
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, 
                                                    shuffle = True, random_state = 43)


# grid / random search에서 사용할 매개 변수                  
# parameters = [                                    
#     {'svm__C':[1, 10, 100, 1000], 'svm__kernel':['linear']},                             # __ : _하나만 하면 에러
#     {'svm__C':[1, 10, 100], 'svm__kernel':['rbf'], 'svm__gamma':[0.001, 0.0001]},   
#     {'svm__C':[1, 100, 1000], 'svm__kernel':['linear'], 'svm__gamma':[0.001, 0.0001]}
# ]    # pipeline에 search을 엮을 경우 parameter에 모델명을 명시 해줘야지 인식한다.

parameters = [
    {'svc__C':[1, 10, 100, 1000], 'svc__kernel':['linear']},                            
    {'svc__C':[1, 10, 100], 'svc__kernel':['rbf'], 'svc__gamma':[0.001, 0.0001]},   
    {'svc__C':[1, 100, 1000], 'svc__kernel':['linear'], 'svc__gamma':[0.001, 0.0001]}
]    # make_pipline : 쓰이는 모델의 풀네임을 써줘야한다(소문자로)

#2. model
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])    
pipe = make_pipeline(MinMaxScaler(), SVC())                  

model = RandomizedSearchCV(pipe, parameters , cv = 5, n_jobs= -1)

#3. fit
model.fit(x_train, y_train)


#4. evaluate, predict
acc = model.score(x_test, y_test)

print('최적의 매개변수 = ', model.best_estimator_)               # 매번 훈련마다 최적의 parameter, acc변화함
print('acc : ', acc)


import sklearn as sk
print('sklearn: ', sk.__version__)                              # sklearn version확인
## make_pipeline은 버전이 0.22.1이어야 함

# acc :  0.9333333333333333
# sklearn:  0.22.1