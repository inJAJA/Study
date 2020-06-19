from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

# parameter
n_estimator = 150         

learning_rate = 0.02          # 경사 하강법에서 얼마나 내려갈(학습할) 것인지 
                              # / 최적의 w를 구하기 위해서 최소의 loss를 구하는 단위
colsample_bytree = 0.9     
colsample_bylevel = 0.7   

max_depth = 5              
n_jobs = -1                

model = XGBClassifier(max_depth = max_depth, learning_rate = learning_rate,
                    n_estimator = n_estimator, n_jobs = n_jobs,
                    colsample_bytree = colsample_bytree,
                    colsample_bylevel = colsample_bylevel)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('점수 :', score)

print(model.feature_importances_)



'''
# learning_rate
: 학습률
: 경사 하강법에서 얼마나 내려갈(학습할) 것인지를 설정함
: 최적의 w를 구하기 위해서 최소의 loss를 구하는 것이 목표
: 너무 줄이면 속도가 느려진다. 
'''

'''
점수 : 0.9666666666666667
[0.16498326 0.04733383 0.33948866 0.4481943 ]
'''