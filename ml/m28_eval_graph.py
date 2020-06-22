from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

x, y = load_boston(return_X_y=True)
print(x.shape)      # (506, 13)
print(y.shape)      # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)

model = XGBRegressor(n_estimators=300, learning_rate=0.1)
                    # = epochs
                                                     # loss
model.fit(x_train, y_train, verbose=True, eval_metric=['logloss','rmse'],
                eval_set=[(x_train, y_train), (x_test, y_test)] # rmse, mae, logloss, error, auc
                # ,early_stopping_rounds= 100
                )


result = model.evals_result()
print("eval's results :", result)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
# print('r2 Score : %.2f%%'%(r2*100.0))
print('r2 :', r2)


epochs = len(result['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['logloss'], label = 'Train')
ax.plot(x_axis, result['validation_1']['logloss'], label = 'Test')
ax.legend()
plt.ylabel('Log loss')
plt.title('XGBoost Log Loss')
# plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['rmse'], label = 'Train')
ax.plot(x_axis, result['validation_1']['rmse'], label = 'Test')
ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost Log Loss')
plt.show()

