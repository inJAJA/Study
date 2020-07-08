import numpy as np
h = np.transpose([[2, 4, 6]])

y1 = np.transpose([[1, 2, 3]])

y2 = np.array([1, 2, 3])

print(h.shape)
print(y1.shape)
print(y2.shape)


print(h - y1)
print(h - y2)