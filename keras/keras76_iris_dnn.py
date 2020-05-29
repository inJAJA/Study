# 다중 분류
from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data
y = iris.target
print(x.shape)      # (150, 4)
print(y.shape)      # (150, )