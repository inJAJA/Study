import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings                                

warnings.filterwarnings('ignore')                            # warnings이라는 에러에 대해서 넘어가겠다.

boston = pd.read_csv('D:/Study/data/csv/boston_house_prices.csv', header = 1)

print(boston)

x = boston.iloc[:, 0:13]
y = boston.iloc[:, 13]

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

allAlegorithms = all_estimators(type_filter = 'regressor')  
               # : sklearn의 모든 regressor의 모델이 저장되어 있음

for (name, algorithm) in allAlegorithms:
    model = algorithm()

    model.fit(x_train, y_train)                                # 훈련
    y_pred = model.predict(x_test)                             # 예측
    print(name, "의 정답률 = ", r2_score(y_test, y_pred)) # acc


import sklearn
print(sklearn.__version__)

'''
Name: MEDV, Length: 506, dtype: float64
ARDRegression 의 정답률 =  0.7385855359771576
AdaBoostRegressor 의 정답률 =  0.863852101905496
BaggingRegressor 의 정답률 =  0.9176423041988245
BayesianRidge 의 정답률 =  0.7489142277563738
CCA 의 정답률 =  0.7749569424747091
DecisionTreeRegressor 의 정답률 =  0.8227261703596701
ElasticNet 의 정답률 =  0.6662534357446657
ElasticNetCV 의 정답률 =  0.6465211400827209
ExtraTreeRegressor 의 정답률 =  0.6874487162110536
ExtraTreesRegressor 의 정답률 =  0.9030772977121497
GaussianProcessRegressor 의 정답률 =  -4.904527258611498
GradientBoostingRegressor 의 정답률 =  0.9248195920925772
HuberRegressor 의 정답률 =  0.5245918525216653
KNeighborsRegressor 의 정답률 =  0.5401612153026705
KernelRidge 의 정답률 =  0.7796746801706546
Lars 의 정답률 =  0.7621351463298275
LarsCV 의 정답률 =  0.7573717946531124
Lasso 의 정답률 =  0.6399927356461492
LassoCV 의 정답률 =  0.6837946514509451
LassoLars 의 정답률 =  -2.7606132582347342e-05
LassoLarsCV 의 정답률 =  0.7627879537708422
LassoLarsIC 의 정답률 =  0.7622134786227752
LinearRegression 의 정답률 =  0.7634174432138485
LinearSVR 의 정답률 =  0.7231220657050819
MLPRegressor 의 정답률 =  0.4097804494656022
'''



