# RandomiziedSearchCV
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#1. data
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, 
                                                    shuffle = True, random_state = 43)


# grid / random search에서 사용할 매개 변수                  
parameters = {                               
    'rf__n_estimators':[100, 200],                             # : 결정트리의 갯수를 지정, defalut = 10
    'rf__max_depth':[1, 3, 5],                                 # : 트리의 최재 깊이, dsfalut = None
    'rf__min_samples_split': [2, 5, 10],                       # : 노드를 분할하기 위한 최소한의 샘플 수, 작을 수록 과적합 가능성 증가
    'rf__min_samples_leaf': [1, 5, 7],                         # : 리프노드가 되기 위해 필요한 최소한의 샘플 개수, 과적합 제어
    'rf__max_features':[2, 'sqrt', 'log2'],  # auto = sqrt     # : 최적의 분할을 위해 고려할 최대 feature개수, default = 'auto'
    # 'rf__max_leaf_nodes' : [],
    # 'rf__min_impurity_decrease':[]
    # 'rf__min_impurity_split': []
}    # pipeline에 search을 엮을 경우 parameter에 모델명을 명시 해줘야지 인식한다.

# parameters = {
#     'randomforestclassifier__n_estimators':[100, 200],
#     'randomforestclassifier__max_depth':[1, 3, 5],
#     'randomforestclassifier__min_samples_split': [2, 5, 10],
#     'randomforestclassifier__min_samples_leaf': [1, 5, 7],
#     'randomforestclassifier__max_features':[2, 'sqrt', 'log2'],  # auto = sqrt
#     # 'randomforestclassifier__max_leaf_nodes' : [],
#     # 'randomforestclassifier__min_impurity_decrease':[]
#     # 'randomforestclassifier__min_impurity_split': []
#     # make_pipline : 쓰이는 모델의 풀네임을 써줘야한다(소문자로)
# }

#2. model
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pipe = Pipeline([("scaler", MinMaxScaler()), ('rf', RandomForestClassifier())])    
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())                  

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

# acc :  1.0
# sklearn:  0.22.1