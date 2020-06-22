from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size = 0.8, random_state = 13
)

model = DecisionTreeClassifier(max_depth =4)      # max_depth 몇 이상 올라가면 구분 잘 못함
                                                  
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print('acc: ', acc)

print(model.feature_importances_)

## feature_importances
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]                                # n_features = column개수 
    plt.barh(np.arange(n_features), model.feature_importances_,      # barh : 가로방향 bar chart
              align = 'center')                                      # align : 정렬 / 'edge' : x축 label이 막대 왼쪽 가장자리에 위치
    plt.yticks(np.arange(n_features), cancer.feature_names)          # tick = 축상의 위치표시 지점
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)                                         # y축의 최솟값, 최댓값을 지정/ x는 xlim

plot_feature_importances_cancer(model)
plt.show()

'''
acc:  0.8859649122807017
[0.         0.04857917 0.         0.         0.01295445 0.
 0.00231786 0.         0.01457182 0.         0.0095245  0.
 0.         0.01251451 0.         0.         0.         0.
 0.         0.         0.78802642 0.         0.         0.
 0.01214479 0.         0.         0.09936649 0.         0.        ]
'''