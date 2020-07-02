import numpy as np
import matplotlib.pyplot as plt

def selu(x, a =1.67326):
    y_list = []
    for x in x:
        if x > 0:
            y = x
        if x <= 0:
            y = a*(np.exp(x) - 1)
        y_list.append(y)
    return y_list

def selu(x, a = 1.6732):
    return list(map(lambda x : x if x > 0 else a*(np.exp(x)-1), x))

x = np.arange(-5, 5, 0.1)
y = selu(x)

plt.plot(x, y)
plt.grid()
plt.show()



