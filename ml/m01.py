import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1)        # 0 ~ 10까지 0.1씩 증가
y = np.sin(x)                    # x에 대한 sin값

plt.plot(x, y)                  

plt.show()
