from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

x, y = load_breast_cancer(return_X_y=True)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBClassifier(n_estimators = 100, learning_rate = 0.05, n_jobs = -1)

model.fit(x_train, y_train, verbose = True, eval_metric= ['logloss', 'auc'],
                            eval_set= [(x_train, y_train),(x_test, y_test)],
                            early_stopping_rounds= 20)

result = model.evals_result()
print("eval's result : ", result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('Accuracy : ', acc)

epochs = len(result['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['logloss'], label = 'Train')
ax.plot(x_axis, result['validation_1']['logloss'], label = 'Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBbosst Log Loss')

ax.plot(x_axis, result['validation_0']['auc'], label = 'Train')
ax.plot(x_axis, result['validation_1']['auc'], label = 'Test')
ax.legend()
plt.ylabel('error')
plt.title('XGBbosst error')

# Accuracy :  0.9649122807017544