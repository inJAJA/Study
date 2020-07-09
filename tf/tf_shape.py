import numpy as np

# y_data를 [None]가 아니라 [None, n]로 써야 하는 이유
h = np.transpose([[2, 4, 6]])

y1 = np.transpose([[1, 2, 3]])

y2 = np.array([1, 2, 3])

print(h.shape)    # (3, 1)
print(y1.shape)   # (3, 1)
print(y2.shape)   # (3,)

print(h - y1)
# [[1]
#  [2]
#  [3]]

print(h - y2)
# [[ 1  0 -1]
#  [ 3  2  1]
#  [ 5  4  3]]

# 두 개의 결과가 다르게 나온다.


#=========================================================================================
import tensorflow as tf
# bias shape 설정법
tf.set_random_seed(777)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]

y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x = tf.placeholder(tf.float32, shape = [None, 4])
y = tf.placeholder(tf.float32, shape = [None, 3])

W = tf.Variable(tf.random_normal([4, 3]), name = 'weigth')
# b = tf.Variable(tf.random_normal([1, 3]), name = 'bias')  # 두 개의 계산이 동일하게 나온다.
b = tf.Variable(tf.random_normal([3]), name = 'bias')

mul = tf.matmul(x, W) 
hypothesis =  mul + b

sess = tf.Session()

sess.run(tf.global_variables_initializer())

x_shape = tf.shape(x)
y_shape = tf.shape(y)
W_shape = tf.shape(W)
b_shape = tf.shape(b)

print(sess.run([x_shape, y_shape, W_shape, b_shape], feed_dict={x:x_data, y:y_data}))

mul_run, h, bias, _ = sess.run([mul, hypothesis, b , W], feed_dict = {x:x_data})
print(mul_run, '\n-----------\n', bias, '\n----------------\n', h)