## 과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. feature수를 줄인다.
# 3. regularization
                
from xgboost import XGBRegressor, plot_importance  
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 회귀 모델
dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

# 트리구조 : 결측치 제거, 데이터 전처리 필요 없음

# parameter
n_estimator = 158          # 나무의 개수 ( 앙상블 )
learning_rate = 0.072      # 학습률 : 이걸 건드는 것이 제일 효과가 크다. / default = 0.01
colsample_bytree = 0.85    # 각 트리마다의 feature 샘플링 비율  : 0.6 ~ 0.9  / max 1  // 샘플 = 전체 나무 ( dropout과 유사 )
colsample_bylevel = 0.6    # 샘플을 어느정도 사용할 것인가.     : 0.6 ~ 0.9 / max 1

max_depth = 5              # 큰 영향을 주지 않는다.
n_jobs = -1                # 딥러닝 아닌 경우에 무조건 ' -1 ' 쓰기 / 딥러닝에서는 잘 터짐

model = XGBRegressor(max_depth = max_depth, learning_rate = learning_rate,
                    n_estimator = n_estimator, n_jobs = n_jobs,
                    colsample_bytree = colsample_bytree,
                    colsample_bylevel = colsample_bylevel)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('점수 :', score)

print(model.feature_importances_)


# feature_importances_ 를 graph로 보여줌
plot_importance(model)
# plt.show()

# f0 : 첫번째 feature  
# fn : n번째 feature

'''
n_estimator = 158         
learning_rate = 0.075       
colsample_bytree = 0.85    
colsample_bylevel = 0.6  

max_depth = 5              
n_jobs = -1   

점수 : 0.9454292908626116
[0.03410036 0.00293195 0.01892266 0.00249341 0.07248564 0.27010834
 0.01231734 0.04208761 0.01143922 0.02806832 0.12431377 0.01252528
 0.36820608]
'''