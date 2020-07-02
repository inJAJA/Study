import numpy as np

x = [0, 0.001, 0.01, 0.1]

log = np.log1p(x)
print(log)            # [0.         0.0009995  0.00995033 0.09531018]

expm = np.expm1(log)
print(expm)           # [0.    0.001 0.01  0.1  ]