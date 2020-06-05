# RandomForest 적용
# cifar10 적용

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score   # Kfold : 교차 검증
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# gridSearch
# 내가 정해놓은 조건들을 충족하는 것을 전부다 가져온다. 


#1. data
cancer = load_breast_cancer()

x = cancer.data         
y = cancer.target

print(x.shape)          # (569, 30)
print(y.shape)          # (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 13, train_size =0.8)

parameters ={
    "n_estimators" : [100, 200],       # 결정 트리의 개수 : default = 10
    "max_depth": [6, 8, 10, 20],       # 트리의 깊이
    "min_samples_leaf":[3, 5, 7, 10],  # 리프노드가 되기 위한 최소한의 샘플 데이터 수 
    "min_samples_split": [2,3, 5],     # node를 분할하기 위한 최소한의 데이터 수
    # 'max_features                    # 데이터의 feature를 참조할 비율 : default = auto / ex) 'auto', 'sqrt', 'log2', 0.25
    "n_jobs" : [-1]                    # 
}                                     

kfold = KFold(n_splits = 5, shuffle = True)                                                  # train, validation

     
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv =  kfold)  
      # default = 10

model.fit(x_train, y_train)

print('최적의 매개변수 : ', model.best_estimator_)
y_pred = model.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred) )


# criterion='gini',
#             max_depth=None, max_features='auto', max_leaf_nodes=None,        
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=5,
#             min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,      
#             oob_score=False, random_state=None, verbose=0,
#             warm_start=False)
# 최종 정답률 =  0.9122807017543859

'''
# GridSearch
 : 일정한 간격으로 점을 정해서 그 점들에 대해 1개씩 차례로 시험해 보면서 최적의 값을 찾음

# RandomSearch
 : 간격이 일정하지 않고 무작위로 최적값은 찾는다.
'''