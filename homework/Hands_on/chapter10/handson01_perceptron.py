import numpy as np
from sklearn.datasets import load_iris 
from sklearn.linear_model import Perceptron

iris = load_iris()
x = iris.data[:, (2, 3)] # 꽃잎의 길이와 너비
y = (iris.target == 0).astype(np.int) # 부채붓꽃(iris setosa)인가?


per_clf = Perceptron()
per_clf.fit(x, y)

y_pred = per_clf.predict([[2, 0.5]])
print(y_pred)                        # [0]