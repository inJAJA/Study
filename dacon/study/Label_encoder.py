import numpy as np
from sklearn.preprocessing import LabelEncoder

x_train = np.array(['PC', 'MOBILE', 'PC'])
x_test = np.array(['PC', 'TABLET', 'MOBILE'])  # x_test에만 TABLET 데이터가 있음

# label encoder 생성
encoder = LabelEncoder()

# x_train 데이터를 이용 피팅하고 라벨숫자로 변환
encoder.fit(x_train)
x_train_encoded = encoder.transform(x_train)

# x_test데이터에만 존재하는 새로 출현한 데이터를 신규 클래스로 추가
for label in np.unique(x_test):
    if label not in encoder.classes_: # unseen label 데이터인 경우
        encoder.classes_ = np.append(encoder.classes_, label) # 미처리시 ValueError

x_test_encoded = encoder.transform(x_test)