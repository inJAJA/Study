from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

x, y = load_iris(return_X_y=True)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBClassifier(objective='multi:softmax', n_estimators = 100, learning_rate = 0.05, n_jobs = -1)
                    # binary:logistic: 이항 분류, logistic regressor모형, 반환값이 클래스가 아니라 예측 확률.
                    # multi:softmax: 다항 분류, Softmax사용 반횐되는 값이 예측확률이 아니라 클래스임. 또한 num_class도 지정해야함.
                    # multi:softprob: 각 클래스 범주에 속하는 예측확률을 반환함.


model.fit(x_train, y_train, verbose = True, eval_metric= ['mlogloss', 'merror'],
                            eval_set= [(x_train, y_train),(x_test, y_test)],
                            early_stopping_rounds= 20)

result = model.evals_result()
print("eval's result : ", result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred.round())
print('Accuracy : ', acc)

epochs = len(result['validation_0']['mlogloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['mlogloss'], label = 'Train')
ax.plot(x_axis, result['validation_1']['mlogloss'], label = 'Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBbosst Log Loss')

ax.plot(x_axis, result['validation_0']['merror'], label = 'Train')
ax.plot(x_axis, result['validation_1']['merror'], label = 'Test')
ax.legend()
plt.ylabel('error')
plt.title('XGBbosst error')

# Accuracy :  1.0