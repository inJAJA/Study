from xgboost import XGBRegressor, plot_importance  
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import numpy as np

# 회귀 모델
x, y = load_boston(return_X_y=True)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = LGBMRegressor(n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 

model.fit(x_train, y_train)

threshold = np.sort(model.feature_importances_)

for thres in threshold:
    selection = SelectFromModel(model, threshold = thres, prefit = True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model = LGBMRegressor(n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 
    
    evals_results = {}

    selection_model.fit(select_x_train, y_train, verbose= False, eval_metric= ['logloss', 'rmse'],
                                        eval_set= [(select_x_train, y_train), (select_x_test, y_test)],
                                        early_stopping_rounds= 20)

    y_pred = selection_model.predict(select_x_test)
    r2 = r2_score(y_test, y_pred)

    print("Thresh=%.3f, n = %d, R2 : %.2f%%" %(thres, select_x_train.shape[1], r2*100.0))

    result = selection_model.evals_result_
    # print("eval's result : ", result)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # pickle로 save
    import pickle                                  # write binary 
    pickle.dump(model, open("./model/xgb_save/boston.LGBM.dat", "wb"))

# Thresh=0.000, n = 13, R2 : 92.13%
# Thresh=6.000, n = 12, R2 : 92.13%
# Thresh=45.000, n = 11, R2 : 92.44%
# Thresh=46.000, n = 10, R2 : 91.90%
# Thresh=88.000, n = 9, R2 : 92.19%
# Thresh=89.000, n = 8, R2 : 90.91%
# Thresh=110.000, n = 7, R2 : 89.99%
# Thresh=147.000, n = 6, R2 : 89.74%
# Thresh=149.000, n = 5, R2 : 87.90%
# Thresh=172.000, n = 4, R2 : 86.15%
# Thresh=204.000, n = 3, R2 : 85.86%
# Thresh=237.000, n = 2, R2 : 83.56%
# Thresh=239.000, n = 1, R2 : 69.52%
