from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import numpy as np

x, y = load_breast_cancer(return_X_y=True)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = LGBMClassifier(n_estimators = 100, learning_rate = 0.05, n_jobs = -1)

model.fit(x_train, y_train)

threshold = np.sort(model.feature_importances_)

for thres in threshold:
    selection = SelectFromModel(model, threshold = thres, prefit = True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model = LGBMClassifier(n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 

    selection_model.fit(select_x_train, y_train, verbose= False, eval_metric= ['logloss', 'error'],
                                        eval_set= [(select_x_train, y_train), (select_x_test, y_test)],
                                        early_stopping_rounds= 20)

    result = selection_model.evals_result_
    # print("eval's result : ", result)

    y_pred = selection_model.predict(select_x_test)
    acc = accuracy_score(y_test, y_pred)
     
    print("Thresh=%.3f, n = %d, ACC : %.2f%%" %(thres, select_x_train.shape[1], acc*100.0))

    # result = selection_model.evals_result()
    # print("eval's result : ", result)

    import pickle                                  # write binary 
    pickle.dump(model, open("./model/xgb_save/cancer.LGBM.dat", "wb"))

# Thresh=6.000, n = 30, ACC : 97.37%
# Thresh=11.000, n = 29, ACC : 97.37%
# Thresh=21.000, n = 28, ACC : 96.49%
# Thresh=22.000, n = 27, ACC : 96.49%
# Thresh=26.000, n = 26, ACC : 96.49%
# Thresh=33.000, n = 25, ACC : 96.49%
# Thresh=34.000, n = 24, ACC : 96.49%
# Thresh=35.000, n = 23, ACC : 96.49%
# Thresh=38.000, n = 22, ACC : 95.61%
# Thresh=43.000, n = 21, ACC : 95.61%
# Thresh=45.000, n = 20, ACC : 95.61%
# Thresh=46.000, n = 19, ACC : 95.61%
# Thresh=47.000, n = 18, ACC : 95.61%
# Thresh=47.000, n = 18, ACC : 95.61%
# Thresh=49.000, n = 16, ACC : 95.61%
# Thresh=50.000, n = 15, ACC : 95.61%
# Thresh=53.000, n = 14, ACC : 95.61%
# Thresh=55.000, n = 13, ACC : 95.61%
# Thresh=56.000, n = 12, ACC : 95.61%
# Thresh=56.000, n = 12, ACC : 95.61%
# Thresh=62.000, n = 10, ACC : 95.61%
# Thresh=70.000, n = 9, ACC : 95.61%
# Thresh=70.000, n = 9, ACC : 95.61%
# Thresh=74.000, n = 7, ACC : 92.98%
# Thresh=79.000, n = 6, ACC : 92.11%
# Thresh=96.000, n = 5, ACC : 92.11%
# Thresh=112.000, n = 4, ACC : 92.11%
# Thresh=126.000, n = 3, ACC : 91.23%
# Thresh=145.000, n = 2, ACC : 91.23%
# Thresh=201.000, n = 1, ACC : 71.93%