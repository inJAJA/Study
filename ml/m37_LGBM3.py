from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import numpy as np
import time
x, y = load_iris(return_X_y=True)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = LGBMClassifier( n_estimators = 100, learning_rate = 0.05, n_jobs = -1)

model.fit(x_train, y_train)

threshold = np.sort(model.feature_importances_)

start = time.time()

for thres in threshold:
    selection = SelectFromModel(model, threshold = thres, prefit = True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model = LGBMClassifier( n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 

    selection_model.fit(select_x_train, y_train, verbose= False, eval_metric= ['multi_logloss', 'multi_error'],
                                        eval_set= [(select_x_train, y_train), (select_x_test, y_test)],
                                        early_stopping_rounds= 20)
    
    result = selection_model.evals_result_
    # print("eval's result : ", result)

    y_pred = selection_model.predict(select_x_test)
    acc = accuracy_score(y_test, y_pred)
     
    print("Thresh=%.3f, n = %d, ACC : %.2f%%" %(thres, select_x_train.shape[1], acc*100.0))

    # result = selection_model.evals_result()
    # print("eval's result : ", result)
    
    import pickle                                 
    pickle.dump(model, open("./model/xgb_save/iris.LGBM.dat", "wb"))
end = time.time() - start
print(" 걸린 시간 :", end) 
# Thresh=98.000, n = 4, ACC : 93.33%
# Thresh=273.000, n = 3, ACC : 93.33%
# Thresh=301.000, n = 2, ACC : 93.33%
# Thresh=327.000, n = 1, ACC : 56.67%
#  걸린 시간 : 0.12276124954223633

# eval_metric 참고 링크
# https://lightgbm.readthedocs.io/en/latest/Parameters.html