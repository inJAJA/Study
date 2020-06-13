import pandas as pd
from pandas import DataFrame

attri_data1 = {'ID':['100','101','102','103','104','106','108','110','111','113'],
               "city": ["서울",'부산','대전','광주','서울','서울','부산','대전','광주','서울'],
               "brith_year":[1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name":['영이','순돌','짱구','태양','션','유리','현아','태식','민수','호식']}

atrri_data_frame1 = DataFrame(attri_data1)


# 분할 리스트 생성
birth_year_bins = [1980, 1985, 1990, 1995, 2000]

''' pd.cut() '''
# 구간 분할 실시
birth_year_cut_data = pd.cut(atrri_data_frame1.brith_year, birth_year_bins)

print(birth_year_cut_data)
# 0    (1985, 1990]
# 1    (1985, 1990]
# 2    (1990, 1995]
# 3    (1995, 2000]
# 4    (1980, 1985]
# 5    (1990, 1995]
# 6    (1985, 1990]
# 7    (1985, 1990]
# 8    (1990, 1995]
# 9    (1980, 1985]
# Name: brith_year, dtype: category
# Categories (4, interval[int64]): [(1980, 1985] < (1985, 1990] < (1990, 1995] < (1995, 2000]]


''' .value_counts() '''
# 각 구간의 수 집계
print(pd.value_counts(birth_year_cut_data))
# (1985, 1990]    4
# (1990, 1995]    3
# (1980, 1985]    2
# (1995, 2000]    1
# Name: brith_year, dtype: int64


''' labels '''
group_names = ['first1980', 'secaond1980', 'first1990', 'second1990']
birth_year_cut_data = pd.cut(atrri_data_frame1.brith_year, birth_year_bins, labels = group_names)
print(pd.value_counts(birth_year_cut_data))
# secaond1980    4
# first1990      3
# first1980      2
# second1990     1
# Name: brith_year, dtype: int64


''' pd.cut( , n) : 분할수 지정 '''
print(pd.cut(atrri_data_frame1.brith_year, 2))
# 0      (1989.0, 1997.0]
# 1    (1980.984, 1989.0]
# 2      (1989.0, 1997.0]
# 3      (1989.0, 1997.0]
# 4    (1980.984, 1989.0]
# 5      (1989.0, 1997.0]
# 6    (1980.984, 1989.0]
# 7      (1989.0, 1997.0]
# 8      (1989.0, 1997.0]
# 9    (1980.984, 1989.0]
# Name: brith_year, dtype: category
# Categories (2, interval[float64]): [(1980.984, 1989.0] < (1989.0, 1997.0]

