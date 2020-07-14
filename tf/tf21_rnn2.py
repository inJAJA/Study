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

x_data = x_data.reshape(5, 1, 5)
y_data = y_data.reshape(5, 1)

print('=================')
print(x_data)
print(y_data)
print('=================')

seq_len = 1
input_dim = 5
batch_size = 5
output = 10
steps = 300
output_dim = 1

X = tf.compat.v1.placeholder(tf.float32, (None, seq_len, input_dim))
Y = tf.compat.v1.placeholder(tf.int32, (None, seq_len))
print(X)                                                            # (?, 1, 5)
print(Y)                                                            # (?, 1)


#2. 모델구성
# model.add(LSTM(output, input_shape = (1, 5)))
# cell = tf.nn.rnn_cell.BasicLSTMCell(output)                       # rnn은 두번 계산됨으로 cell을 준비해줌
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
                            # model.add(LSTM)
print(hypothesis)                                                   # (?, 1, 11)

''' < 추정 > : 아마도 분류형 loss '''
#3-1. 컴파일
weights = tf.ones([batch_size, seq_len])                                       
sequence_loss = tf.contrib.seq2seq.sequence_loss(                   
    logits = hypothesis, targets = Y, weights = weights )           
    #        y_pred            y_true                               
    # [batch_size, sequence_length, num_classes] /[batch_size, sequence_length]  
    #               (?, 1, 11)                   / (5, 1)                                                              

# node sequence_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
# Y를 알아서 one-hot-encoding하여 인식
# Y의 가장 큰 값이 10이기 때문에 원핫인코딩하면 컬럼이 11개로 된다.
# -> output = 11이상 부터 가능하다 
#    / 10이하면 [ ERROR : Received a label value of 10 which is outside ]


cost = tf.reduce_mean(sequence_loss)                                # sequence_loss의 전체 평균

# train = tf.train.AdamOptimizer(learning_rate=1e-1).minimize(loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-1).minimize(cost)

prediction = tf.argmax(hypothesis, axis = 2)

#3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(steps):
        loss, _ = sess.run([cost, train], feed_dict = {X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})
        print(step, 'loss :', loss, 'prediction :', result.reshape(-1), 'true Y :', y_data.reshape(-1))
        # print(sess.run(hypothesis, feed_dict = {X:x_data}))


