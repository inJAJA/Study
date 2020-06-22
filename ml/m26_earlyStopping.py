from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

x, y = load_boston(return_X_y=True)
print(x.shape)      # (506, 13)
print(y.shape)      # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)

model = XGBRegressor(n_estimators=400, learning_rate=0.1)

model.fit(x_train, y_train, verbose=True, eval_metric='rmse',
                eval_set=[(x_train, y_train), (x_test, y_test)], # rmse, mae, logloss, error, auc
                early_stopping_rounds= 20)
                # 꺽인 시점으로 반환(20번 흔들리면 흔들리기 시작하는 시점)

result = model.evals_result()
print("eval's results :", result)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
# print('r2 Score : %.2f%%'%(r2*100.0))
print('r2 :', r2)
'''
[98]    validation_0-rmse:0.43465       validation_1-rmse:2.37777      
Stopping. Best iteration:
[78]    validation_0-rmse:0.57088       validation_1-rmse:2.36899 
'''