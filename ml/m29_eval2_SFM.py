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

    selection_model.fit(x_train, y_train, verbose= True, eval_metric= ['logloss', 'error'],
                                        eval_set= [(x_train, y_train), (x_test, y_test)],
                                        early_stopping_rounds= 20)

    y_pred = selection_model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print('acc : ', acc)
     
    print("Thresh=%.3f, n = %d, ACC : %.2f%%" %(thres, select_x_train.shape[1], acc*100.0))

    # result = selection_model.evals_result()
    # print("eval's result : ", result)

    # acc :  0.9649122807017544

