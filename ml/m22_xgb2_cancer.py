from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

breast_cancer = load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

# parameter
n_estimator = 200          
learning_rate = 0.08       
colsample_bytree = 0.85   
colsample_bylevel = 0.6   

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
n_estimator = 200          
learning_rate = 0.08       
colsample_bytree = 0.85   
colsample_bylevel = 0.6   

max_depth = 5              
n_jobs = -1 

점수 : 0.9736842105263158
[0.05175772 0.02395859 0.00616497 0.0034043  0.0065834  0.00621285
 0.03317067 0.14424753 0.00454786 0.00303592 0.00851963 0.01620645
 0.01186227 0.01593337 0.00492775 0.01003323 0.01523688 0.00723088
 0.00664227 0.00601987 0.1576953  0.02124699 0.12573738 0.15399826
 0.0204933  0.00489866 0.05002265 0.06397256 0.00812076 0.00811769]
'''