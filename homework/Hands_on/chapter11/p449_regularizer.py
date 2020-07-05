# over_fitting 방지
from tensorflow import keras

# l2 규제
layer = keras.layers.Dense(100, activation = 'elu',
                            kernel_initializer = 'he_normal',
                            kernel_regularizer = keras.regularizers.l2(0.01))


from functools import partial
RegularizedDense = partial(keras.layers.Dense,
                            activaion = 'elu',
                            kernel_initializer = 'he_normal',
                            kernel_regularizer = keras.regularizers.l2(0.01))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    RegularizedDense(300),
    RegularizedDense(100),
    RegularizedDense(10, activation = 'softmax',
                        kernel_initializer = 'glorot_uniform')
])