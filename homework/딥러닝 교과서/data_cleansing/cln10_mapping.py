import pandas as pd
from pandas import DataFrame

attri_data1 = {'ID':['100','101','102','103','104','106','108','110','111','113'],
               "city": ["서울",'부산','대전','광주','서울','서울','부산','대전','광주','서울'],
               "brith_day":[1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name":['영이','순돌','짱구','태양','션','유리','현아','태식','민수','호식']}

atrri_data_frame1 = DataFrame(attri_data1)

print(atrri_data_frame1)
#     ID city  brith_day name
# 0  100   서울       1990   영이
# 1  101   부산       1989   순돌
# 2  102   대전       1992   짱구
# 3  103   광주       1997   태양
# 4  104   서울       1982    션
# 5  106   서울       1991   유리
# 6  108   부산       1988   현아
# 7  110   대전       1990   태식
# 8  111   광주       1995   민수
# 9  113   서울       1981   호식


city_map = {'서울':'서울',
            '광주': '전라도',
            '부산': '경상도',
            '대전':'충청도'}

print(city_map)
# {'서울': '서울', '광주': '전라도', '부산': '경상도', '대전': '충청도'}


''' mapping '''
# 새로운 컬럼 region을 추가합니다. 해당 데이터가 없는 경우 NaN
atrri_data_frame1['region'] = atrri_data_frame1['city'].map(city_map)
print(atrri_data_frame1)
#     ID city  brith_day name region
# 0  100   서울       1990   영이     서울
# 1  101   부산       1989   순돌    경상도
# 2  102   대전       1992   짱구    충청도
# 3  103   광주       1997   태양    전라도
# 4  104   서울       1982    션     서울
# 5  106   서울       1991   유리     서울
# 6  108   부산       1988   현아    경상도
# 7  110   대전       1990   태식    충청도
# 8  111   광주       1995   민수    전라도
# 9  113   서울       1981   호식     서울


# 문제
MS_map = {'서울':'중부',
        '광주': '남부',
        '부산': '남부',
        '대전':'중부'}

atrri_data_frame1['MS'] = atrri_data_frame1['city'].map(MS_map)
print(atrri_data_frame1)
#     ID city  brith_day name region  MS
# 0  100   서울       1990   영이     서울  중부
# 1  101   부산       1989   순돌    경상도  남부
# 2  102   대전       1992   짱구    충청도  중부
# 3  103   광주       1997   태양    전라도  남부
# 4  104   서울       1982    션     서울  중부
# 5  106   서울       1991   유리     서울  중부
# 6  108   부산       1988   현아    경상도  남부
# 7  110   대전       1990   태식    충청도  중부
# 8  111   광주       1995   민수    전라도  남부
# 9  113   서울       1981   호식     서울  중부