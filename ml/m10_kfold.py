import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score   # Kfold : 교차 검증
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings                                

warnings.filterwarnings('ignore')                            # warnings이라는 에러에 대해서 넘어가겠다.


#1. data
iris = pd.read_csv('D:/Study/data/csv/iris.csv', header = 0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

print(x)
print(y)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

kfold = KFold(n_splits = 5, shuffle = True)  # n_splits: 전체 데이터를 n개씩 자르겠다.
                                             # KFold   : 모든 데이터가 훈련에 사용될 수 있도로 n번 학습시킨다.
                                             # 새로운 데이터(다른 범위의 x값들)를 n번 학습시킨다.

allAlegorithms = all_estimators(type_filter = 'classifier')  
               # : sklearn의 모든 classifier의 모델이 저장되어 있음

for (name, algorithm) in allAlegorithms:
    model = algorithm()
    
    scores = cross_val_score(model, x, y, cv=kfold)     # train과 test로 짤리지 않은 5개의 데이터의 점수 계산
                                              
    print(name, '의 정답률 :')
    print(scores)

import sklearn
print(sklearn.__version__)


