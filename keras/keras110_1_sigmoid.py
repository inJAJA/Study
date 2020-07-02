
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):                  # 0과 1 사이로 만들어 줌
    return 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

print(x.shape, y.shape)

plt.plot(x, y)
plt.grid()
plt.show()

# activation : 가중치를 규격화 해줌