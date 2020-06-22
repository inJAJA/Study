'''
m28_eval
1. eval 에  'loss'와 다른 지표 1개 더 추가.
2. earlyStopping 적용
3. plot으로 그릴 것

m29_eval
SelectFromModel에 
1. 회귀
2. 이진 분류
3. 다중 분류

1. eval 에  'loss'와 다른 지표 1개 더 추가.
2. earlyStopping 적용
3. plot으로 그릴 것

4. 결과는 주석으로 소스 하단에 표시.

5. m27 ~ 29까지 완벽 이해할 것!
'''
from xgboost import XGBRegressor, plot_importance  
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 회귀 모델
x, y = load_boston(return_X_y=True)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBRegressor(n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 

model.fit(x_train, y_train, verbose= True, eval_metric= ['logloss', 'rmse'],
                            eval_set= [(x_train, y_train), (x_test, y_test)],
                            early_stopping_rounds= 20)

result = model.evals_result()
print("eval's result : ", result)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('R2 : ', r2)

epochs = len(result['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['logloss'], label = 'Train')
ax.plot(x_axis, result['validation_1']['logloss'], label = 'Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBbosst Log Loss')

ax.plot(x_axis, result['validation_0']['rmse'], label = 'Train')
ax.plot(x_axis, result['validation_1']['rmse'], label = 'Test')
ax.legend()
plt.ylabel('RMSE')
plt.title('XGBbosst RMSE')

# R2 :  0.9354279986548603