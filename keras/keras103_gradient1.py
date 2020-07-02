import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6
x = np.linspace(-1, 6, 100)
y = f(x)

# 그리잣!
plt.plot(x, y, 'k-')  # 'k-' : 줄 긋는다.
plt.plot(2, 2, 'sk')  # 'sk' : 2, 2지점에 점을 찍는다.
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 2차함수의 기울기(미분 값)이 최저가 되는 지점 = loss가 0인 지점