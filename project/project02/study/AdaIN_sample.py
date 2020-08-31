import cv2
from keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py

# load Image
f = h5py.File('D:/data/HDF5/face_Doberman.hdf5', 'r')   # Dog Image
x = f['doberman'][10]
# x = cv2.imread('D:\image/image4.jpg')
# x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
x = cv2.resize(x, dsize = (256, 256), interpolation = cv2.INTER_LINEAR)#/255.
x = x.reshape(-1, 256, 256, 3)

y = cv2.imread('D:/data/Gan/FFHQ/02000/02010.png')      # Human Image
y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
y = cv2.resize(y, dsize=(256, 256), interpolation = cv2.INTER_LINEAR)/ 255.
y = y.reshape(-1, 256, 256, 3)

# keras backend를 사용하기 위해서 numpy -> tensor로 변환
x = tf.convert_to_tensor(x, np.float32)
y = tf.convert_to_tensor(y, np.float32)

axis = -1
epsilon=1e-3

input_shape = K.int_shape(x)
# print(input_shape)               # (400, 750, 3)

reduction_axes = list(range(0, len(input_shape)))
print(reduction_axes)              # [0, 1, 2]

if axis is not None:
    del reduction_axes[axis]

del reduction_axes[0]

print(reduction_axes)

gamma = K.std(y, reduction_axes, keepdims=True) + epsilon   # S(y)
beta = K.mean(y, reduction_axes, keepdims=True)             # mean(y)
print(beta.shape)                   # (1, 1, 1, 3)
print(gamma.shape)                  # (1, 1, 1, 3)

# Adain
mean = K.mean(x, reduction_axes, keepdims=True)             # mean(x)
stddev = K.std(x, reduction_axes, keepdims=True) + epsilon  # S(x)
normed = (x - mean) / stddev

print(mean.shape)                   # (1, 1, 1, 3)
print(stddev.shape)                 # (1, 1, 1, 3)
print(normed.shape)                 # (1, 400, 750, 3)

# result
image = normed * gamma + beta
print(image.shape)                  # (1, 400, 750, 3)


normed = K.reshape(normed, (256, 256, 3))

x = K.reshape(x, (256, 256, 3))
y = K.reshape(y, (256, 256, 3))
image = K.reshape(image, (256, 256, 3))

# show Image
fig, axes = plt.subplots(1, 3)

axes[0].imshow(x)
axes[1].imshow(y)
axes[2].imshow(image)

axes[0].set_title('Contents Image')
axes[1].set_title('Style Image')
axes[2].set_title('Mix_Style')

axes[0].axis('off')
axes[1].axis('off')
axes[2].axis('off')

plt.show()