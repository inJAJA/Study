import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

# 인공지능 겨울 극복 : layer를 여러개 쌓아서 선을 여러개 그어준다.

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])


# 첫번째 layer                        # output node = 3
w1 = tf.Variable(tf.random_normal([2, 100]), name = 'weight1')
b1 = tf.Variable(tf.random_normal([100]), name = 'bias1')
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)
# model.add(Dense(100, input_dim = 2))

# 두번째 layer                        
w2 = tf.Variable(tf.random_normal([100, 50]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([50]), name = 'bias1')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)
# model.add(Dense(50, input_dim = 100))

# 세번째 layer
w3 = tf.Variable(tf.random_normal([50, 1]), name = 'weight3')
b3 = tf.Variable(tf.random_normal([1]), name = 'bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)                  # 마지막 output_layer
# model.add(Dense(1, input_dim = 50))


cost =  -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 4.9e-2)
train = optimizer.minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _= sess.run([cost, train], feed_dict = {x: x_data, y: y_data})
    
        if step % 10 == 0 :
            print(step, 'cost :',cost_val)
    
    h, c, a = sess.run([hypothesis, predict, accuracy], feed_dict={x:x_data, y:y_data})
    print('\n Hypothesis :\n', h, '\n Correct (y) :\n', c, 
          '\n Accuracy :', a)
