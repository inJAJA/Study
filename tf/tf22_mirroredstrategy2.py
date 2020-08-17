import tensorflow as tf
import numpy as np
import os

# data\
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[..., None] / 255.
x_test = x_test[..., None] / 255.


# strategy
strategy = tf.distribute.MirroredStrategy()
print('장치의 수: {}'.format(strategy.num_replicas_in_sync))


BUFFER_SIZE = len(x_train)

BATCH_SIZE_PER_REPLICA = 64                                                 # GPU하나당 입력
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync  # 전체 입력



train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset =tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)


# model
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation = 'relu', input_shape = (28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation = 'relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])

    model.compile(loss = 'sparse_categorical_crossentropy',
                    optimizer = tf.keras.optimizers.Adam(),
                    metrics=['acc'])

model.fit(train_dataset, epochs = 12)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

eval_loss, eval_acc = model.evaluate(eval_dataset)

print('loss = {}, acc = {}'.format(eval_loss, eval_acc))