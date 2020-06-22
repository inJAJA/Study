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
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# 회귀 모델
x, y = load_boston(return_X_y=True)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBRegressor(n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 

model.fit(x_train, y_train)

threshold = np.sort(model.feature_importances_)

for thres in threshold:
    selection = SelectFromModel(model, threshold = thres, prefit = True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model = XGBRegressor(n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 

    selection_model.fit(select_x_train, y_train, verbose= False, eval_metric= ['logloss', 'rmse'],
                                        eval_set= [(select_x_train, y_train), (select_x_test, y_test)],
                                        early_stopping_rounds= 20)

    y_pred = selection_model.predict(select_x_test)
    r2 = r2_score(y_test, y_pred)

    print("Thresh=%.3f, n = %d, R2 : %.2f%%" %(thres, select_x_train.shape[1], r2*100.0))

    # result = selection_model.evals_result()
    # print("eval's result : ", result)

# Thresh=0.003, n = 13, R2 : 93.54%
# Thresh=0.005, n = 12, R2 : 93.71%
# Thresh=0.006, n = 11, R2 : 93.69%
# Thresh=0.009, n = 10, R2 : 93.78%
# Thresh=0.012, n = 9, R2 : 94.11%
# Thresh=0.014, n = 8, R2 : 94.31%
# Thresh=0.015, n = 7, R2 : 93.76%
# Thresh=0.017, n = 6, R2 : 92.80%
# Thresh=0.017, n = 5, R2 : 93.63%
# Thresh=0.039, n = 4, R2 : 92.26%
# Thresh=0.045, n = 3, R2 : 89.30%
# Thresh=0.248, n = 2, R2 : 81.05%
# Thresh=0.569, n = 1, R2 : 69.21%