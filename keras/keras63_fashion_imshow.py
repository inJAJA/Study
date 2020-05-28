import numpy as np
import matplotlib.pyplot as plt

# 과제 1 
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


plt.imshow(x_train[0])
plt.show()

