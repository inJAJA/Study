import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings                                

warnings.filterwarnings('ignore')                            # warnings이라는 에러에 대해서 넘어가겠다.

iris = pd.read_csv('D:/Study/data/csv/iris.csv', header = 0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

allAlegorithms = all_estimators(type_filter = 'classifier')  
               # : sklearn의 모든 classifier의 모델이 저장되어 있음

for (name, algorithm) in allAlegorithms:
    model = algorithm()

    model.fit(x_train, y_train)                                # 훈련
    y_pred = model.predict(x_test)                             # 예측
    print(name, "의 정답률 = ", accuracy_score(y_test, y_pred)) # acc


import sklearn
print(sklearn.__version__)

'''
AdaBoostClassifier 의 정답률 =  0.9666666666666667
BaggingClassifier 의 정답률 =  0.9666666666666667
BernoulliNB 의 정답률 =  0.2
CalibratedClassifierCV 의 정답률 =  0.8333333333333334
ComplementNB 의 정답률 =  0.5666666666666667
DecisionTreeClassifier 의 정답률 =  0.9666666666666667
ExtraTreeClassifier 의 정답률 =  0.9333333333333333
ExtraTreesClassifier 의 정답률 =  0.9666666666666667
GaussianNB 의 정답률 =  0.9666666666666667
GaussianProcessClassifier 의 정답률 =  0.9666666666666667
GradientBoostingClassifier 의 정답률 =  0.9666666666666667        
KNeighborsClassifier 의 정답률 =  1.0
LabelPropagation 의 정답률 =  1.0
LabelSpreading 의 정답률 =  1.0
LinearDiscriminantAnalysis 의 정답률 =  1.0
LinearSVC 의 정답률 =  0.9
LogisticRegression 의 정답률 =  0.8333333333333334                # regression이라고 적혀있지만 분류 모델이다
LogisticRegressionCV 의 정답률 =  0.8
MLPClassifier 의 정답률 =  0.9333333333333333
MultinomialNB 의 정답률 =  0.5666666666666667
NearestCentroid 의 정답률 =  0.9666666666666667
NuSVC 의 정답률 =  0.9666666666666667
PassiveAggressiveClassifier 의 정답률 =  0.5666666666666667       
Perceptron 의 정답률 =  0.5666666666666667
QuadraticDiscriminantAnalysis 의 정답률 =  1.0
RadiusNeighborsClassifier 의 정답률 =  0.9666666666666667
RandomForestClassifier 의 정답률 =  0.9666666666666667
RidgeClassifier 의 정답률 =  0.7666666666666667
RidgeClassifierCV 의 정답률 =  0.7666666666666667
SGDClassifier 의 정답률 =  0.5666666666666667
SVC 의 정답률 =  0.9666666666666667
'''



