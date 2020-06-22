from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score

x, y = load_breast_cancer(return_X_y=True)
print(x.shape)      
print(y.shape)     

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)

model = XGBClassifier(n_estimators=1000, learning_rate=0.1)
              
model.fit(x_train, y_train, verbose=True, eval_metric='error',
                eval_set=[(x_train, y_train), (x_test, y_test)])


result = model.evals_result()
# print("eval's results :", result)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred) 
print('acc : ', acc)

#############################################################################

''' save''' 
import pickle # python제공                                   # write b
pickle.dump(model, open("./model/xgb_save/cancer.pickle.dat", "wb")) # model을 cancer.pickle.dat에 넣겠다.

print("저장됬다.")
# acc :  0.9736842105263158
# 저장됬다.


''' load '''                                                   # read
model2 = pickle.load(open("./model/xgb_save/cancer.pickle.dat", "rb"))
print("불러왔다.")

y_pred = model2.predict(x_test)
acc = accuracy_score(y_test, y_pred) 
print('acc : ', acc)
# 불러왔다.
# acc :  0.9736842105263158