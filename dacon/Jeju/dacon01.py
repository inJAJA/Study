import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
import time

def grap_year(data):
    data = str(data)
    return int(data[:4])

def grap_month(data):
    data = str(data)
    return int(data[4:])

start = time.time()

data = pd.read_csv('./data/dacon/Jeju/201901-202003.csv')
print(data.info())
#  #   Column        Dtype
# ---  ------        -----
#  0   REG_YYMM      int64    년월
#  1   CARD_SIDO_NM  object   카드이용지역_시도
#  2   CARD_CCG_NM   object   카드이용지역_시군구(가맹점 주소 기준)
#  3   STD_CLSS_NM   object   업종명
#  4   HOM_SIDO_NM   object   거주지역_시도(고객 집주소 기준)
#  5   HOM_CCG_NM    object   거주지역_시군구(고객 집주소 기준)
#  6   AGE           object   연령대
#  7   SEX_CTGO_CD   int64    성별(1: 남성, 2: 여성)
#  8   FLC           int64    가구생애주기 (1: 1인가구, 2: 영유아자녀가구, 3: 중고생자녀가구, 4: 성인자녀가구, 5: 노년가구)
#  9   CSTMR_CNT     int64    이용고객수 (명)
#  10  AMT           int64    이용금액 (원)
#  11  CNT           int64    이용건수 (건)
# dtypes: int64(6), object(6)
# memory usage: 2.2+ GB
print(data.isnull().sum())
# REG_YYMM             0
# CARD_SIDO_NM         0
# CARD_CCG_NM      87213
# STD_CLSS_NM          0
# HOM_SIDO_NM          0
# HOM_CCG_NM      147787
# AGE                  0
# SEX_CTGO_CD          0
# FLC                  0
# CSTMR_CNT            0
# AMT                  0
# CNT                  0
# dtype: int64

data =data.fillna('')
data['year'] = data['REG_YYMM'].apply(lambda x: grap_year(x))     # year month
data['month'] = data['REG_YYMM'].apply(lambda x: grap_month(x))   # 2020    03
data = data.drop(['REG_YYMM'], axis = 1)

# 데이터 정제
df = data.copy()
df = df.drop(['CARD_CCG_NM', 'HOM_CCG_NM'], axis=1)

columns = ['CARD_SIDO_NM', 'STD_CLSS_NM', 'HOM_SIDO_NM', 'AGE', 'SEX_CTGO_CD', 'FLC', 'year', 'month']
df = df.groupby(columns).sum().reset_index(drop = False)

# 인코딩
dtypes = df.dtypes
encoders = {}
for column in df.columns:
    if str(dtypes[column]) == 'object':
        encoder = LabelEncoder()
        encoder.fit(df[column])
        encoders[column] = encoder
        
df_num = df.copy()        
for column in encoders.keys():
    encoder = encoders[column]
    df_num[column] = encoder.transform(df[column])

print(df_num['AGE'])
'''
# feature, target설정
train_num = df_num.sample(frac=1, random_state=0)
train_features = train_num.drop(['CSTMR_CNT', 'AMT', 'CNT'], axis=1)
train_target = np.log1p(train_num['AMT'])

# 훈련
model = XGBRegressor(n_jobs=-1, random_state=0)
# model = MultiOutputRegressor(xgb)
model.fit(train_features, train_target)

# 예측 템플릿 만들기
CARD_SIDO_NMs = df_num['CARD_SIDO_NM'].unique()
STD_CLSS_NMs  = df_num['STD_CLSS_NM'].unique()
HOM_SIDO_NMs  = df_num['HOM_SIDO_NM'].unique()
AGEs          = df_num['AGE'].unique()
SEX_CTGO_CDs  = df_num['SEX_CTGO_CD'].unique()
FLCs          = df_num['FLC'].unique()
years         = [2020]
months        = [4, 7]

temp = []
for CARD_SIDO_NM in CARD_SIDO_NMs:
    for STD_CLSS_NM in STD_CLSS_NMs:
        for HOM_SIDO_NM in HOM_SIDO_NMs:
            for AGE in AGEs:
                for SEX_CTGO_CD in SEX_CTGO_CDs:
                    for FLC in FLCs:
                        for year in years:
                            for month in months:
                                temp.append([CARD_SIDO_NM, STD_CLSS_NM, HOM_SIDO_NM, AGE, SEX_CTGO_CD, FLC, year, month])
temp = np.array(temp)
temp = pd.DataFrame(data=temp, columns=train_features.columns)

# 예측
pred = model.predict(temp)
pred = np.expm1(pred)
temp['AMT'] = np.round(pred, 0)
temp['REG_YYMM'] = temp['year']*100 + temp['month']
temp = temp[['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]
temp = temp.groupby(['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM']).sum().reset_index(drop=False)

# 디코딩 
temp['CARD_SIDO_NM'] = encoders['CARD_SIDO_NM'].inverse_transform(temp['CARD_SIDO_NM'])
temp['STD_CLSS_NM'] = encoders['STD_CLSS_NM'].inverse_transform(temp['STD_CLSS_NM'])

# 제출 파일 만들기
submission = pd.read_csv('./data/dacon/Jeju/submission.csv', index_col=0)
submission = submission.drop(['AMT'], axis=1)
submission = submission.merge(temp, left_on=['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM'], right_on=['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM'], how='left')
submission.index.name = 'id'
submission.to_csv('submission.csv', encoding='utf-8-sig')
submission.head()

end = time.time() - start
print('END : ',end)
'''