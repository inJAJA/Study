import os
root_logdir = os.path.join(os.curdir, 'my_logs')

def get_run_logdir():
    import time
    run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir() # ex) './my_logs/run_2019_06_07-15_15_22'


[...] # 모델 구성과 컴파일
from tensorflow import keras
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(x_train, y_train, epochs = 30,
                    validation_data = (x_val, y_val),
                    callbacks = [tensorboard_cb])


# create_file_writer()함수를 사용해  SummartWriter를 만들고
# with블럭 안에서 tensorboard를 사용해 시각화할 수 이는 scalar, histogram, images, texts, audio를 기록
import tensorflow as tf
import numpy as np
test_logdir = get_run_logdir()
writer = tf.summary.create_file_writer(test_logdir)
with writer.as_defalut():
    for step in range(1, 1000 +1):
        tf.summary.scalar('my_scaler', np.sin(step /10), step = step)
        data = (np.random.randn(100) + 2) * step / 100 # 몇몇 랜덤 data
        tf.summary.histogram('my_hist', data, buckets = 50, step=step)
        images = np.random.rand(2, 32, 32, 3) # 32x32 RGB 이미지
        texts = ['The step is' + str(step), 'Its square is '+ str(step**2)]
        tf.summary.text('my_text', texts, step=step)
        sine_wave = tf.math.sin(tf.range(12000)/48000 * 2 * np.pi * step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
        tf.summary.audio('my_audio', audio, sample_rate=48000, step=step) 