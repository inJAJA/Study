from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np


cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size = 0.8, random_state = 22
)


# model = DecisionTreeClassifier(max_depth =4)      # max_depth : 5이상이 되면 과적합 가능성 다수
model = RandomForestClassifier()                    # tree구조의 model은 과적합이 잘 일어난다.

# max_features : 기본값 써라
# n_estimatior : 클수록 좋다 / 단점 : 메모리 많이 차지, 기본값 = 100
# n_jobs = -1  : 병렬처리

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
acc:  0.956140350877193
[0.02671389 0.01596345 0.051907   0.06007211 0.00375716 0.00510971
 0.03933072 0.11877758 0.00243109 0.0027776  0.00784257 0.00445432
 0.00944499 0.02824439 0.00365302 0.0038348  0.00401625 0.00838701
 0.00497142 0.00642498 0.13109913 0.02419484 0.13422562 0.12272713
 0.01095635 0.00936416 0.02050104 0.11606862 0.014653   0.00809605]
'''