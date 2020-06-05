import pandas as pd
import matplotlib.pyplot as plt

# 와인 데이터 읽기
wine = pd.read_csv('D:/Study/data/csv/winequality-white.csv', header = 0, sep =  ';')

y = wine['quality']
x = wine.drop('quality', axis =1)                 # drop을 사용하려면 header가 필요하다

print(x.shape)
print(y.shape)

# y 레이블 축소 : binning 
newlist = []
for i in list(y) : 
    if i <= 4 :
        newlist += [0]
    elif i <= 7:
        newlist += [1]
    else:
        newlist += [2]

y = newlist

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print('acc_score: ', accuracy_score(y_test, y_pred))
print('acc: ', acc)

# acc_score:  0.9418367346938775
# acc:  0.9418367346938775