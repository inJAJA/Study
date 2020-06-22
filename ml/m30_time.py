from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score, accuracy_score

x, y = load_boston(return_X_y=True)                    # x, y가 그냥 들어간다.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("R2 : ", score)

''' Start_1 '''
import time
start = time.time()

# feature Engineering
thresholds = np.sort(model.feature_importances_)      # 오름차순 정렬

print(thresholds)

for thresh in thresholds:                             # 전체 컬럼 수만큼 돈다.
    selection = SelectFromModel(model, threshold= thresh, prefit = True)
                                                # median
    select_x_train = selection.transform(x_train)     # for문을 돌릴 때 마다 컬럼이 하나씩 빠진다.(중요도가 낮은 것 부터)
    print(select_x_train.shape)                      

    selection_model = XGBRegressor(n_estimators = 1000)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred) 
    # print('R2 :', score)

    print("Thresh=%.3f, n = %d, R2 : %.2f%%" %(thresh, select_x_train.shape[1],
                                                score*100.0))

''' END '''



''' Start_2 '''
start2 = time.time()

# feature Engineering
thresholds = np.sort(model.feature_importances_)      # 오름차순 정렬

print(thresholds)

for thresh in thresholds:                             # 전체 컬럼 수만큼 돈다.
    selection = SelectFromModel(model, threshold= thresh, prefit = True)
                                                # median
    select_x_train = selection.transform(x_train)     # for문을 돌릴 때 마다 컬럼이 하나씩 빠진다.(중요도가 낮은 것 부터)
    print(select_x_train.shape)                      

    selection_model = XGBRegressor( n_jobs = -1, n_estimators = 1000)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred) 
    # print('R2 :', score)

    print("Thresh=%.3f, n = %d, R2 : %.2f%%" %(thresh, select_x_train.shape[1],
                                                score*100.0))

''' END '''
end = start2 - start
print("그냥 걸린 시간 : ", end) 
end2 = time.time() - start2
print("n_jobs 걸린 시간 : ", end2) 

# 그냥 걸린 시간 :  4.683350563049316
# n_jobs 걸린 시간 :  4.53106427192688