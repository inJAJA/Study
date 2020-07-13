import tensorflow as tf
import numpy as np

# data = hihello

idx2char = ['e','h','i','l','o']

_data = np.array([['h','i','h','e','l','l','o']], dtype=np.str).reshape(-1, 1)
print(_data.shape)                                  # (7, 1)
print(_data)                                        # [['h'] ['i'] ['h'] ['e'] ['l'] ['l'] ['o']]
print(type(_data))                                  # <class 'numpy.ndarray'>

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()

print('====================')
print(_data)
                                                    # [[0. 1. 0. 0. 0.]
                                                    #  [0. 0. 1. 0. 0.]
                                                    #  [0. 1. 0. 0. 0.]
                                                    #  [1. 0. 0. 0. 0.]
                                                    #  [0. 0. 0. 1. 0.]
                                                    #  [0. 0. 0. 1. 0.]
                                                    #  [0. 0. 0. 0. 1.]]
print(type(_data))                                  # <class 'numpy.ndarray'>
print(_data.dtype)                                  # float64


x_data = _data[:6, ]
y_data = _data[1:, ]

print('========== x =========')
print(x_data)                                       # (6, 5)
print('========== y =========')
print(y_data)                                       # (6, 5)
print('======================')

y_data = np.argmax(y_data, axis = 1)                # shape를 맞춰주기 위해서
print('====== y argmax ======')
print(y_data)                                       # [2 1 0 3 3 4]
print(y_data.shape)                                 # (6, )

x_data = x_data.reshape(1, 6, 5)
y_data = y_data.reshape(1, 6)

print(x_data.shape)                                 # (1, 6, 5)
print(y_data.shape)                                 # (1, 6)

sequence_len = 6
input_dim =5
output = 5
batch_size = 1                                      # 전체 행

# X = tf.placeholder(tf.float32, (None, sequence_len, input_dim))
# Y = tf.placeholder(tf.float32, (None, sequence_len))
X = tf.compat.v1.placeholder(tf.float32, (None, sequence_len, input_dim))
Y = tf.compat.v1.placeholder(tf.int32, (None, sequence_len))
print(X)    # Tensor("Placeholder:0", shape=(?, 6, 5), dtype=float32)
print(Y)    # Tensor("Placeholder_1:0", shape=(?, 6), dtype=float32)


#2. 모델구성
# model.add(LSTM(output, input_shape = (6, 5)))
# cell = tf.nn.rnn_cell.BasicLSTMCell(output)                          # rnn은 두번 계산됨으로 cell을 준비해줌
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
                            # model.add(LSTM)
print(hypothesis)                                                    # (?, 6, 5)

#3-1. 컴파일
weights = tf.ones([batch_size, sequence_len])                                        
sequence_loss = tf.contrib.seq2seq.sequence_loss(                   # sequence형 loss가 따로 있다.
    logits = hypothesis, targets = Y, weights = weights )
    #        y_pred            y_true
cost = tf.reduce_mean(sequence_loss)                                # sequence_loss의 전체 평균

# train = tf.train.AdamOptimizer(learning_rate=1e-1).minimize(loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-1).minimize(cost)

prediction = tf.argmax(hypothesis, axis = 2)

#3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(401):
        loss, _ = sess.run([cost, train], feed_dict = {X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})
        print(step, 'loss :', loss, 'prediction :', result, 'true Y', y_data)
        print(sess.run(hypothesis, feed_dict = {X:x_data}))

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\nPrediction str :", ''.join(result_str))
