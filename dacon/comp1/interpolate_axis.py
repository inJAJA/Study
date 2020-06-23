import numpy as np
import pandas as pd

#  get some noised linear data
df = pd.DataFrame({"A":[1, 2, 3, None, 5], 
                   "B":[None, 12, 13, 14, None], 
                   "C":[21, 22, None, 24, 25], 
                   "D":[31, 32, None, None, 35]}) 
print(df)
#      A     B     C     D
# 0  1.0   NaN  21.0  31.0
# 1  2.0  12.0  22.0  32.0
# 2  3.0  13.0   NaN   NaN
# 3  NaN  14.0  24.0   NaN
# 4  5.0   NaN  25.0  35.0

df0 = df.interpolate(axis = 0)
print(df0)
#      A     B     C     D
# 0  1.0   NaN  21.0  31.0
# 1  2.0  12.0  22.0  32.0
# 2  3.0  13.0  23.0  33.0
# 3  4.0  14.0  24.0  34.0
# 4  5.0  14.0  25.0  35.0
''' 열 보간 '''

df1 = df.interpolate(axis = 1)
print(df1)
#      A     B     C     D
# 0  1.0  11.0  21.0  31.0
# 1  2.0  12.0  22.0  32.0
# 2  3.0  13.0  13.0  13.0
# 3  NaN  14.0  24.0  24.0
# 4  5.0  15.0  25.0  35.0
''' 행 보간 '''
