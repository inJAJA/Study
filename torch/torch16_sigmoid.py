
import numpy as np
import matplotlib.pyplot as plt

# sigmoid 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, 'g')
plt.plot([0, 0], [1.0, 0.0], ":")   # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()

# W값에 따른 그래프 변화 : y = W * x + b
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5 * x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

plt.plot(x, y1, 'r', linestyle = '--')  # W = 0.5
plt.plot(x, y2, 'g')                    # W = 1
plt.plot(x, y3, 'b', linestyle = '--')  # W = 2
plt.plot([0, 0], [1.0, 0.0], ':')       # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()                              
                                        # => W값이 작아지면 경사가 작아짐

# b값에 따른 그래프 변화
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x+0.5)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--')    # b = 0.5
plt.plot(x, y2, 'g')                    # b = 1
plt.plot(x, y3, 'b', linestyle='--')    # b = 1.5
plt.plot([0,0],[1.0,0.0], ':')          # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()
                                        # => b가 크면 좌측으로, 작으면 우측으로 이동 (1 기준)
                                        