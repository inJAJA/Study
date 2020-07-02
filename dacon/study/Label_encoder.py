import numpy as np
from sklearn.preprocessing import LabelEncoder

x_train = np.array(['PC', 'MOBILE', 'PC'])
x_test = np.array(['PC', 'TABLET', 'MOBILE'])  # x_test에만 TABLET 데이터가 있음

# label encoder 생성
encoder = LabelEncoder()

encoder.fit(x_train)                           # x_train 데이터를 이용 피팅하고 라벨숫자로 변환
x_train_encoded = encoder.transform(x_train)


for label in np.unique(x_test):                # x_test데이터에만 존재하는 새로 출현한 데이터를 신규 클래스로 추가
    if label not in encoder.classes_:          # unseen label 데이터인 경우
        encoder.classes_ = np.append(encoder.classes_, label) # 미처리시 ValueError

x_test_encoded = encoder.transform(x_test)


dtypes = df.dtypes
print(dytypes)                                 # 각 컬럼의 dtype
encoders = {}
for column in df.columns:                      # fit
    if str(dtypes[column]) == 'object':        # dataframe에서 dtype이 object인 것들만 걸러서 인코딩
        encoder = LabelEncoder()
        encoder.fit(df[column])
        encoders[column] = encoder
        
df_num = df.copy()        
for column in encoders.keys():                  # transform
    encoder = encoders[column]
    df_num[column] = encoder.transform(df[column])