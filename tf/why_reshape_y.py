import numpy as np
h = np.transpose([[2, 4, 6]])

y1 = np.transpose([[1, 2, 3]])

y2 = np.array([1, 2, 3])

print(h.shape)    # (3, 1)
print(y1.shape)   # (3, 1)
print(y2.shape)   # (3,)


print(h - y1)
# [[1]
#  [2]
#  [3]]
print(h - y2)
# [[ 1  0 -1]
#  [ 3  2  1]
#  [ 5  4  3]]