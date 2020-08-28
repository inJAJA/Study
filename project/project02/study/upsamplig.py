from keras.models import Sequential, Model
from keras.layers import Input, UpSampling2D

from keras.datasets import mnist


(x_train, y_train),(_, _) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)

x = Input(shape = (28, 28, 1))
up = UpSampling2D()(x)

model = Model(x, up)

model.summary()
'''
Model: "model_1"
_________________________________________________________________       
Layer (type)                 Output Shape              Param #
=================================================================       
input_1 (InputLayer)         (None, 28, 28, 1)         0
_________________________________________________________________       
up_sampling2d_1 (UpSampling2 (None, 56, 56, 1)         0
=================================================================       
Total params: 0
Trainable params: 0
Non-trainable params: 0
_________________________________________________________________ 
'''

x_pred = model.predict(x_train[0][None, ...])
print(x_pred.shape)

# image
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(x_train[0].reshape(28, 28))
ax1.set_title('origin 28x28')
ax1.axis([0, 55, 55, 0])

ax2.imshow(x_pred.reshape(56, 56))
ax2.set_title('Upsampling 56x56')
ax2.axis([0, 55, 55, 0])

plt.show()