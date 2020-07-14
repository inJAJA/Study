import tensorflow as tf
import numpy as np

dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(dataset.shape)                                    # (10, )

# RNN 모델을 짜시오!

def xy_split(dataset, step):
    x_data = []
    y_data = []
    for i in range(len(dataset)-step):
        start = i
        end = i + step 
        x = dataset[start : end]
        y = dataset[end]
        x_data.append(x)
        y_data.append(y)
    return np.array(x_data), np.array(y_data)

x_data, y_data = xy_split(dataset, 5)
print(x_data.shape, x_data) # (5, 5)
print(y_data.shape, y_data) # (5, )

x_data = x_data.reshape(5, 5, 1)
y_data = y_data.reshape(5, 1)

print('=================')
print(x_data)
print(y_data)
print('=================')

seq_len = 5
input_dim = 1
batch_size = 5
output = 100
steps = 300
output_dim = 1

X = tf.compat.v1.placeholder(tf.float32, (None, seq_len, input_dim))
Y = tf.compat.v1.placeholder(tf.float32, (None, 1))
print(X)                                                            # (?, 5, 1)
print(Y)                                                            # (?, 1)

# rnn출력을 Fully connected를 한 번 더 거쳐 출력해주겠다. 
# 이때 몇개로 펼쳐줄지는 hidden_dim에 값을 넣어서 조정해준다.

#2. 모델구성
# model.add(LSTM(output, input_shape = (1, 5)))
cell = tf.nn.rnn_cell.BasicLSTMCell(output)                       # rnn은 두번 계산됨으로 cell을 준비해줌
# cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
                            # model.add(LSTM)
print(hypothesis)                                                   # (?, 1, 11)

''' 회귀 loss 사용 '''
#3-1-2. 컴파일
Y_pred = tf.contrib.layers.fully_connected(
    hypothesis[:, -1], output_dim, activation_fn=None)
    # We use the last cell's output. 
    # 예를들어 1일차부터 5일차까지 출력을 다 쓰는게 아니라 5일까지 모은 데이터의 출력(가장 마지막 출력)을 쓰는 것이므로
    # 3차원 -> 2차원
    # [참고] https://jfun.tistory.com/194

cost = tf.reduce_sum(tf.square(Y_pred - Y)) # sum of the squares

train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-1).minimize(cost)


#3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(steps):
        loss, _ = sess.run([cost, train], feed_dict = {X:x_data, Y:y_data})
        result = sess.run(Y_pred, feed_dict={X:x_data})
        print(step, 'loss :', loss, '\nprediction :', result.reshape(-1), '\ntrue Y :', y_data.reshape(-1))
        # print(sess.run(hypothesis, feed_dict = {X:x_data}))
        print('-------------------------------------------------------')