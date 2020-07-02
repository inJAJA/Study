import keras.backend as K
import numpy as np
import tensorflow as tf

x = np.array(range(10))
print(x)                   # [0 1 2 3 4 5 6 7 8 9]

# square
square = K.square(x)       # x : Tensor or variable
                           # return : a tensor
print(square)              # tf.Tensor([ 0  1  4  9 16 25 36 49 64 81], shape=(10,), dtype=int32)

x1 = np.array([0, -1, -2, -3, 4, 5, -7, -8, 9, 10])
abs = K.abs(x1)
print(abs)                 # tf.Tensor([ 0  1  2  3  4  5  7  8  9 10], shape=(10,), dtype=int32)

