import pandas as pd
import matplotlib.pyplot as plt

# 와인 데이터 읽기
wine = pd.read_csv('D:/Study/data/csv/winequality-white.csv', header = 0, sep =  ';')

count_data = wine.groupby('quality')['quality'].count()

print(count_data)
# quality
# 3      20
# 4     163
# 5    1457
# 6    2198                               
# 7     880
# 8     175
# 9       5
# Name: quality, dtype: int64

count_data.plot()
plt.show()
'''
# 한 곳에 집중되어 분포도가 떨어짐
# y값이 5, 6로 집중되어 머신이 학습할 때 5, 6로 예측하면 대부분의 경우는 맞출 수 있지만
# 다른 값들에 대해서는 정확도가 떨어진다         -> 일정 정확도 이상으로 올리기 힘들다.

# 분류되는 종류을 좁혀줘서 분포도를 어느정도 맞춰주면 해결할 수 있다.
# [3, 4, 5, 6, 7, 8, 9] -> [0, 1, 2]

# BUT! y값의 컬럼을 바꾸는 것은 권한이 있을 때 해야한다.
#      : 의뢰자가 원하는 방향으로 값이 나와야하기 때문에  
'''