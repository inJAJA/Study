import pandas as pd

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header =None)
# 각 수치가 무엇을 나타내는지 컬럼 헤더를 추가합니다.
df.columns =[ "", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Mangnesium", 
            "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", 
            "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]


# 평균값 구하기
print(df['Alcohol'].mean())      # 13.000617977528083

# 문제
print(df["Mangnesium"].mean())   # 99.74157303370787

