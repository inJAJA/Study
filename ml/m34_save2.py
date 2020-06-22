from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

x, y = load_breast_cancer(return_X_y=True)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBClassifier(n_estimators = 100, learning_rate = 0.05, n_jobs = -1)

model.fit(x_train, y_train)

threshold = np.sort(model.feature_importances_)

for thres in threshold:
    selection = SelectFromModel(model, threshold = thres, prefit = True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model = XGBClassifier(n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 

    selection_model.fit(select_x_train, y_train, verbose= False, eval_metric= ['logloss', 'error'],
                                        eval_set= [(select_x_train, y_train), (select_x_test, y_test)],
                                        early_stopping_rounds= 20)

    y_pred = selection_model.predict(select_x_test)
    acc = accuracy_score(y_test, y_pred)
     
    print("Thresh=%.3f, n = %d, ACC : %.2f%%" %(thres, select_x_train.shape[1], acc*100.0))

    # result = selection_model.evals_result()
    # print("eval's result : ", result)

    model.save_model("./model/xgb_save/breast_cancer_thresh=%.3f-acc=%.2f.model"%(thres, acc))

# Thresh=0.001, n = 30, ACC : 96.49%
# Thresh=0.002, n = 29, ACC : 96.49%
# Thresh=0.002, n = 28, ACC : 96.49%
# Thresh=0.003, n = 27, ACC : 96.49%
# Thresh=0.004, n = 26, ACC : 96.49%
# Thresh=0.004, n = 25, ACC : 96.49%
# Thresh=0.004, n = 24, ACC : 96.49%
# Thresh=0.004, n = 23, ACC : 96.49%
# Thresh=0.005, n = 22, ACC : 96.49%
# Thresh=0.005, n = 21, ACC : 96.49%
# Thresh=0.006, n = 20, ACC : 96.49%
# Thresh=0.006, n = 19, ACC : 96.49%
# Thresh=0.008, n = 18, ACC : 96.49%
# Thresh=0.008, n = 17, ACC : 96.49%
# Thresh=0.008, n = 16, ACC : 96.49%
# Thresh=0.010, n = 15, ACC : 96.49%
# Thresh=0.011, n = 14, ACC : 96.49%
# Thresh=0.014, n = 13, ACC : 96.49%
# Thresh=0.015, n = 12, ACC : 96.49%
# Thresh=0.017, n = 11, ACC : 96.49%
# Thresh=0.018, n = 10, ACC : 96.49%
# Thresh=0.019, n = 9, ACC : 96.49%
# Thresh=0.020, n = 8, ACC : 96.49%
# Thresh=0.026, n = 7, ACC : 96.49%
# Thresh=0.032, n = 6, ACC : 97.37%
# Thresh=0.082, n = 5, ACC : 95.61%
# Thresh=0.110, n = 4, ACC : 94.74%
# Thresh=0.123, n = 3, ACC : 96.49%
# Thresh=0.166, n = 2, ACC : 92.11%
# Thresh=0.267, n = 1, ACC : 88.60%