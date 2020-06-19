# 1. 분류
from sklearn.svm import LinearSVC                     # 선형 분류    : 직선만 가능
from sklearn.svm import SVC                           # 다항식 커널

from sklearn.neighbors import KNeighborsClassifier    # 군집 분석

from sklearn.ensemble import RandomForestClassifier


# 2. 회귀
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor


# booster
from xgboost import XGBRegressor, XGBClassifier, plot_importance  