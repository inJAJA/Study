import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


#1. data
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, 
                                                    shuffle = True, random_state = 43)

#2. model
# model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
                                                                  # 전처리와 model이 한번에 돌아감
# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])     # 쓸 scaler와 model을 명시
pipe = make_pipeline(MinMaxScaler(), SVC())                       # '이름'써줄 필요 없음

pipe.fit(x_train, y_train)

print('acc: ', pipe.score(x_test, y_test))
# acc:  1.0