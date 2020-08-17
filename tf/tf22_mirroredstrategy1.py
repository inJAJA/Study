'''
# tf.distribute.Strategy
: 훈련을 여러 GPU 또는 여러 장비, 여러 TPU로 나우어 처리하기 위한 tensorflow API
    => 분산 처리

# tf.distribute.MirroredStrategy
: 장비 하나에서 다중 GPU를 이용한 동기 분산 훈련 지원
: 각각의 GPU장치 마다 복제본 만들어짐 
  -> 모델의 모든 변수가 복제본마다 미러링 됨
    -> 변수들은 동일한 변경사항이 함께 적용되어 모두 같은 값을 유지
'''

import tensorflow as tf
import numpy as np
import os

# data
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[..., None]/np.float32(255)  
x_test = x_test[..., None]/np.float32(255)
print(x_train.shape)          # (60000, 28, 28, 1)


# MirroredStrategy()
strategy = tf.distribute.MirroredStrategy()
print('장치의 수: {}'.format(strategy.num_replicas_in_sync))


BUFFER_SIZE = len(x_train)

BATCH_SIZE_PER_REPLICA = 64                                                 # GPU하나당 입력
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync  # 전체 입력

epochs = 10


# 분산 데이터 strategy.scope 내에 생성
with strategy.scope():
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
  train_dist_dataset  = strategy.experimental_distribute_dataset(train_dataset)

  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(GLOBAL_BATCH_SIZE)
  test_dist_dataset  = strategy.experimental_distribute_dataset(test_dataset)


# model
def create_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation = 'relu', input_shape = (28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
  ])

  return model


# checkpoint
checkpoint_dir = './model'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')


# loss
with strategy.scope():
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    reduction = tf.keras.losses.Reduction.NONE)
    # 또는 loss_fn = tf.keras.losses.spare_categorical_crossentropy 사용 가능
  def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions) 
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size = GLOBAL_BATCH_SIZE)
          # 샘플당 손실과 선택적으로 샘플 가중리,  GLOBAL_BATCH_SIZE를 매개변수 값으로 받고 스케일이 조정된 손실 반환

# metrics
with strategy.scope():
  test_loss = tf.keras.metrics.Mean(name = 'test_loss')

  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name = 'train_accuracy'
  )
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name = 'test_accuracy'
  )

# train
with strategy.scope():
  model = create_model()

  optimizer = tf.keras.optimizers.Adam()

  checkpoint = tf.train.Checkpoint(optimizer = optimizer, model = model)


with strategy.scope():
  def train_step(inputs):
    images, labels = inputs

    with tf.GradientTape() as tape:
      predictions = model(images, training=True)
      loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss 

  def test_step(inputs):
    images, labels = inputs

    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)


with strategy.scope():
  # experimental_run_v2 : 주어진 계산 복사, 분산된 입력으로 계산 수행

  @tf.function
  def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.experimental_run_v2(train_step, 
                                                      args =(dataset_inputs, ))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis = None)

  @tf.function
  def distributed_test_step(dataset_inputs):
    return strategy.experimental_run_v2(test_step, args = (dataset_inputs, ))

  for epoch in range(epochs):
    # Train 루프
    total_loss = 0.0
    num_batches = 0
    for x in train_dist_dataset:
      total_loss += distributed_train_step(x)
      num_batches += 1
    train_loss = total_loss / num_batches

    # Test 루프
    for x in test_dist_dataset:
      distributed_test_step(x)

    if epoch % 2 == 0:
      checkpoint.save(checkpoint_prefix)

    template = ("에포크 {}, 손실: {}, 정확도: {}, 테스트 손실: {}, "
                "테스트 정확도: {}")
    print(template.format(epoch+1, train_loss,
                           train_accuracy.result()*100, test_loss.result(),
                           test_accuracy.result()*100))

    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()