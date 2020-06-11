from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
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

train_gap = np.load('./dacon/comp1/train_gap.npy')
test_gap = np.load('./dacon/comp1/test_gap.npy')

x = np.hstack((x, train_gap))
x_pred = np.hstack((x_pred, test_gap))


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state =33)

xgb = XGBRegressor() 

model = MultiOutputRegressor(xgb)

model.fit(x_train, y_train)

y_pred1 = model.predict(x_test)

print('mae: ', mean_absolute_error(y_test, y_pred1))


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


y_predict = model.predict(x_pred)

plot_feature_importances(model)

# submission
a = np.arange(10000,20000)
submission = pd.DataFrame(y_predict, a)
submission.to_csv('./dacon/comp1/sub_XG_gap.csv',index = True, header=['hhb','hbo2','ca','na'],index_label='id')



'''
# mae:  1.4406767626497146
'''