'''
SELU > ELU > LeakyReLU(그리고 변종들) > ReLU > tanh > 로지스틱
'''
from tensorflow import keras

# LeakyReLU 사용법
model = keras.models.Sequential([
    [...],
    keras.layers.Dense(10, kernel_initializer = 'he_normal'),
    keras.layers.LeakyReLU(alpha = 0.2)
    [...]
])

# SELU 사용법
layer = keras.layers.Dense(10, activation = 'selu',
                            kernel_initializer = 'lecun_normal')