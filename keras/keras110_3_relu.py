import numpy as np
import matplotlib.pyplot as plt

def relu(x):                   # x < 0 | x = 0
    return np.maximum(0, x)    # x > 0 | x = x

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()