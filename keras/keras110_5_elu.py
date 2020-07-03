import numpy as np
import matplotlib.pyplot as plt
from keras.activations import elu

'''
def elu(x, a=1):
    y_list = []
    for x in x:
        if x > 0 :
            y = x
        if x<= 0 :
            y = a*(np.exp(x)-1)
        y_list.append(y)
    return y_list
'''
def elu(x, a =1):
    return list(map(lambda x : x if x > 0 else a*(np.exp(x)-1), x))

x = np.arange(-5, 5, 0.1)
y = elu(x)

plt.plot(x, y)
plt.grid()
plt.show()




