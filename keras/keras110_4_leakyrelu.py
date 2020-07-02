import numpy as np
import matplotlib.pyplot as plt

def leakyrelu(x, a = 0.01):                       # x < 0.01*x | x = 0.01*x
    return np.maximum(a*x, x)        # x > 0.01*x | x = x

x = np.arange(-5, 5, 0.1)
y = leakyrelu(x)

plt.plot(x, y)
plt.grid()
plt.show()



