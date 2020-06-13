import pandas as pd
from pandas import Series, DataFrame

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


attri_data2 = {'ID': ['107','109'],
               'city':['봉화','전주'],
               'brith_day':[1994, 1988]}
attri_data_frame2 = DataFrame(attri_data2)

atrri_data_frame1.append(attri_data_frame2).sort_values(by ='ID', ascending = True).reset_index(drop=True)
print(attri_data_frame2)
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
#     ID city  brith_day
# 0  107   봉화       1994
# 1  109   전주       1988