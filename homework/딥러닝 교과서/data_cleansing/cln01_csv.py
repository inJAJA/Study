import pandas as pd

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header =None)
# 각 수치가 무엇을 나타내는지 컬럼 헤더를 추가합니다.
df.columns =[ "", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Mangnesium", 
            "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", 
            "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

print(df)
#         Alcohol  ...  OD280/OD315 of diluted wines  Proline
# 0    1    14.23  ...                          3.92     1065
# 1    1    13.20  ...                          3.40     1050
# 2    1    13.16  ...                          3.17     1185
# 3    1    14.37  ...                          3.45     1480
# 4    1    13.24  ...                          2.93      735
# ..  ..      ...  ...                           ...      ...
# 173  3    13.71  ...                          1.74      740
# 174  3    13.40  ...                          1.56      750
# 175  3    13.27  ...                          1.56      835
# 176  3    13.17  ...                          1.62      840
# 177  3    14.13  ...                          1.60      560

# [178 rows x 14 columns]


# 문제
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header =None)
df.colums = ["sepal length", "sepal width", "petal length", "petal width", 'class']
print(df)