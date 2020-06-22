from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import numpy as np


cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size = 0.8, random_state = 22
)


# model = DecisionTreeClassifier(max_depth =4)      # max_depth : 5이상이 되면 과적합 가능성 다수
# model = RandomForestClassifier()                  # tree구조의 model은 과적합이 잘 일어난다.
model = GradientBoostingClassifier()

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
[2.67784234e-04 3.13365264e-02 9.73309273e-04 8.87592048e-04
 2.71570717e-04 1.45990237e-03 1.45115002e-03 9.43625749e-02
 8.69137114e-06 2.36100244e-05 5.71713621e-03 3.45809617e-05
 4.84405185e-03 7.39779455e-03 6.31380468e-04 3.89189444e-03
 5.31210556e-04 1.09167185e-02 1.24913734e-03 4.26344982e-03
 1.98531206e-01 5.57121980e-02 4.51701355e-01 4.23921339e-02
 9.35846887e-03 1.54682621e-04 1.20691726e-02 5.93039935e-02
 1.32410462e-04 1.24313378e-04]
'''