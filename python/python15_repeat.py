import numpy as np

a = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
print(a)         # [[1 2]
                 #  [3 4]]
print(a.shape)   # (2, 2)


b = np.repeat(a[:,:, np.newaxis], 3, axis = 2)
print(b)        # [[[1 1 1]
                #   [2 2 2]
                #   [3 3 3]]

                #  [[4 4 4]
                #   [5 5 5]
                #   [6 6 6]]

                #  [[7 7 7]
                #   [8 8 8]
                #   [9 9 9]]]
print(b.shape)  # (3, 3, 3)


c = np.repeat(a[..., np.newaxis], 3, -1)
print(c)        # [[[1 1 1]
                #   [2 2 2]
                #   [3 3 3]]

                #  [[4 4 4]
                #   [5 5 5]
                #   [6 6 6]]

                #  [[7 7 7]
                #   [8 8 8]
                #   [9 9 9]]]
print(c.shape)  # (3, 3, 3)

