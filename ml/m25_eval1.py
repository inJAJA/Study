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
                #   ( = 나무의 개수 = epochs)

model.fit(x_train, y_train, verbose=True, eval_metric='rmse',
                eval_set=[(x_train, y_train), (x_test, y_test)])
# rmse, mae, logloss, error, auc
# merror, mlogloss : multiclass

result = model.evals_result()
print("eval's results :", result)


'''
        (x_train, y_train)              (x_test, y_test)
[0]     validation_0-rmse:21.58494      validation_1-rmse:21.68460
'''