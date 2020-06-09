from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

# making TimeSeries with missing values
datestrs = ['6/1/2020', '6/3/2020', '6/4/2020', '6/8/2020','6/10/2020']
dates = pd.to_datetime(datestrs)
print(dates)
print('===============')

ts = Series([1, np.nan, np.nan, 8, 10], index = dates)
print(ts)
# 2020-06-01     1.0
# 2020-06-03     NaN
# 2020-06-04     NaN
# 2020-06-08     8.0
# 2020-06-10    10.0
# dtype: float64

ts_intp_linear = ts.interpolate()                        # 선형 보간법
print(ts_intp_linear)
# 2020-06-01     1.000000
# 2020-06-03     3.333333
# 2020-06-04     5.666667
# 2020-06-08     8.000000
# 2020-06-10    10.000000
# dtype: float64