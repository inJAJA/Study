import cv2
from keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = cv2.imread('D:\image/image4.jpg')
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
x = cv2.resize(x, dsize = (750, 400), interpolation = cv2.INTER_LINEAR)/255.

y = cv2.imread('D:/image/image3.jpg')
y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
y = cv2.resize(y, dsize=(750, 400), interpolation = cv2.INTER_LINEAR)/ 255.

# keras backend를 사용하기 위해서 numpy -> tensor로 변환
x = tf.convert_to_tensor(x, np.float32)
y = tf.convert_to_tensor(y, np.float32)

axis = -1
epsilon=1e-3

input_shape = K.int_shape(x)
# print(input_shape)              # (400, 750, 3)

reduction_axes = list(range(0, len(input_shape)))
print(reduction_axes)              # [0, 1, 2]

beta = y
gamma = y

if axis is not None:
    del reduction_axes[axis]

del reduction_axes[0]

# Adain
mean = K.mean(x, reduction_axes, keepdims=True)
stddev = K.std(x, reduction_axes, keepdims=True) + epsilon
normed = (x - mean) / stddev

# result
image = normed * gamma + beta
print(image.shape)

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 3)

axes[0].imshow(x)
axes[1].imshow(y)
axes[2].imshow(image)

axes[0].set_title('Contents Image')
axes[1].set_title('Style Image')
axes[2].set_title('Mix_Style')

plt.show()