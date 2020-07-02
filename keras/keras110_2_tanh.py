import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)              # -1과 1 사이에 수렴

plt.plot(x, y)
plt.grid()
plt.show()