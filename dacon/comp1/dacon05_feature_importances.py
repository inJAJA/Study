from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

train = pd.read_csv('./data/dacon/comp1/train.csv', index_col= 0 , header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= 0 , header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col= 0 , header = 0)

print('train.shape: ', train.shape)              # (10000, 75)  = x_train, test
print('test.shape: ', test.shape)                # (10000, 71)  = x_predict
print('submission.shape: ', submission.shape)    # (10000, 4)   = y_predict
                    

train = train.interpolate()                       
test = test.interpolate()

x_data = train.iloc[:, :71]                           
y_data = train.iloc[:, -4:]


x_data = x_data.fillna(x_data.mean())
test = test.fillna(test.mean())


x = x_data.values
y = y_data.values
x_pred = test.values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state =33)

model = DecisionTreeRegressor(max_depth =4)                     # max_depth 몇 이상 올라가면 구분 잘 못함
# model = RandomForestRegressor(n_estimators = 200, max_depth=3)
# model = GradientBoostingRegressor()
# model = XGBRegressor()

## feature_importances
def plot_feature_importances(model):
    plt.figure(figsize= (10, 40))
    n_features = x_data.shape[1]                                # n_features = column개수 
    plt.barh(np.arange(n_features), model.feature_importances_,      # barh : 가로방향 bar chart
              align = 'center')                                      # align : 정렬 / 'edge' : x축 label이 막대 왼쪽 가장자리에 위치
    plt.yticks(np.arange(n_features), x_data.columns)          # tick = 축상의 위치표시 지점
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)             # y축의 최솟값, 최댓값을 지정/ x는 xlim
    plt.show()

def tree_fit(y_train, y_test):
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print('score: ', score)
    y_predict = model.predict(x_pred)
    y_pred1 = model.predict(x_test)
    print('mae: ', mean_absolute_error(y_test, y_pred1))
    
    # feature_importance  
    plot_feature_importances(model)
    print(model.feature_importances_)

    return y_predict


def boost_fit_acc(y_train, y_test):
    y_predict = []
    for i in range(len(submission.columns)):
       print(i)
       y_train1 = y_train[:, i]  
       model.fit(x_train, y_train1)
       
       y_test1 = y_test[:, i]
       score = model.score(x_test, y_test1)
       print('score: ', score)

       y_pred = model.predict(x_pred)
       y_pred1 = model.predict(x_test)
       print('mae: ', mean_absolute_error(y_test1, y_pred1))

       # feature_importance
       plot_feature_importances(model)
       print(model.feature_importances_)

       y_predict.append(y_pred)     
    return np.array(y_predict)

y_predict = tree_fit(y_train, y_test)
# y_predict = boost_fit_acc(y_train, y_test).reshape(-1, 4) 

print(y_predict.shape)


# submission
a = np.arange(10000,20000)
submission = pd.DataFrame(y_predict, a)
submission.to_csv('D:/Study/dacon/comp1/sub_XG.csv',index = True, header=['hhb','hbo2','ca','na'],index_label='id')










'''
1. DT
        score:  -0.00880395871833086
          mae:  1.7829054749478388

2. RF
        score:  -0.0017898286593774721
          mae:  1.7761615417568033

3. GB
   col1: score:  -0.015442049646700484
           mae:  2.375662736487169
   col2: score:  -0.011072060845875864
           mae:  0.8305348071167891
   col3: score:  -0.007488559349809742
           mae:  2.401697259378463
   col4: score:  -0.013504983740235676
           mae:  1.531545143515503

4. XGB
   col1: score:  0.6665033334670867
           mae:  1.3308124556362626
   col2: score:  0.12635585858329645
           mae:  0.7698395311427116
   col3: score:  0.1383898538013727
           mae:  2.1989057760238646
   col4: score:  0.05151398280090491
           mae:  1.4631492877960204
    
    제출 : 3.6708596143
'''