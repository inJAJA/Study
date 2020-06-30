import numpy as np

a = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
print(a)        # [[1 2]
                #  [3 4]]
print(a.shape)  # (2, 2)


b = a[np.newaxis, : ,:]
# b = a[np.newaxis, ...]       
print(b)        # [[[1 2 3]
                #   [4 5 6]
                #   [7 8 9]]]
print(b.shape)  # (1, 3, 3)


c = a[:, np.newaxis, :]
print(c)        # [[[1 2 3]]

                #  [[4 5 6]]

                #  [[7 8 9]]]
print(c.shape)  # (3, 1, 3)


c = a[:, :, np.newaxis]
# c = a[..., np.newaxis]       
print(c)        # [[[1]
                #   [2]
                #   [3]]

                #  [[4]
                #   [5]
                #   [6]]

                #  [[7]
                #   [8]
                #   [9]]]
print(c.shape)  # (3, 3, 1)

