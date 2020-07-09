# preprocesing

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def min_max_scaler(dataset):
    numerator = dataset - np.min(dataset, 0)                # 그 열에서 최솟값
    denominator = np.max(dataset, 0) - np.min(dataset, 0)   # 최댓값 - 최솟값
    return numerator / (denominator + 1e-5)                 # 0이 되는 것을 방지하기 위해서 1e-7을 더함

dataset = np.array(
    [
        [828.659973, 833.450012, 908100, 828.349976, 831.659973],

        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],

        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],

        [816, 820.958984, 1008100, 815.48999, 819.23999],

        [819.359985, 823, 1188100, 818.469971, 818.97998],

        [819, 823, 1198100, 816, 820.450012],

        [811.700012, 815.25, 1098100, 809.780029, 813.669983],

        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)

y_data = dataset[:, [-1]]

dataset = min_max_scaler(dataset)
print(dataset)

x_data = dataset[:, 0:-1]
# y_data = dataset[:, [-1]]

print(x_data.shape)
print(y_data.shape)

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 회귀

# x, y, w, b, hypothesis, cost, trian(optimizer)
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([4, 1]), name = 'weight')
                                # x의 열의 값과 동일 해야 함(행렬 계산) 

b = tf.Variable(tf.random_normal([1]), name = 'bias') 

hypothesis = tf.matmul(x, w) + b                      # wx + b (행렬 곱)

cost =  tf.reduce_mean(tf.square( hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1.5e-1) # 0.00001

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _= sess.run([cost, hypothesis, train],
                                feed_dict = {x: x_data, y: y_data})

    # if step % 20 == 0 :
        # print(step, 'cost :',cost_val, '\n 예측값 :', hy_val)

y_pred = sess.run(hypothesis, feed_dict={x:x_data})
print(y_pred)

r2 = r2_score(y_data, y_pred)
print("R2 :", r2)