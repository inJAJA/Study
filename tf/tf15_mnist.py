import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.datasets import mnist          
import numpy as np               
mnist.load_data()                                         

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)   # (60000, 28, 28)
print(y_train.shape)   # (60000,)  

x_train = x_train.reshape(-1, 28*28).astype('float32')/255.
x_test = x_test.reshape(-1, 28*28).astype('float32')/255.


x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 10])


# 1                           # input / output   
w1 = tf.Variable(tf.random_normal([784, 8]), name = 'weight1')
b1 = tf.Variable(tf.random_normal([8]), name = 'bias1')
layer1 = tf.matmul(x, w1) + b1

# 2                       
w2 = tf.Variable(tf.random_normal([8, 16]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([16]), name = 'bias1')
layer2 = tf.matmul(layer1, w2) + b2

# 3                       
w3 = tf.Variable(tf.random_normal([16, 32]), name = 'weight2')
b3 = tf.Variable(tf.random_normal([32]), name = 'bias1')
layer3 = tf.matmul(layer2, w3) + b3

# 4                       
w4 = tf.Variable(tf.random_normal([32, 64]), name = 'weight2')
b4 = tf.Variable(tf.random_normal([64]), name = 'bias1')
layer4 = tf.matmul(layer3, w4) + b4

# 5                       
w5 = tf.Variable(tf.random_normal([64, 128]), name = 'weight2')
b5 = tf.Variable(tf.random_normal([128]), name = 'bias1')
layer5 = tf.matmul(layer4, w5) + b5

# 6                       
w6 = tf.Variable(tf.zeros([128, 258]), name = 'weight2')
b6 = tf.Variable(tf.zeros([258]), name = 'bias1')
layer6 = tf.matmul(layer5, w6) + b6

# 7                       
w7 = tf.Variable(tf.zeros([258, 128]), name = 'weight2')
b7 = tf.Variable(tf.zeros([128]), name = 'bias1')
layer7 = tf.matmul(layer6, w7) + b7

# 8                       
w8 = tf.Variable(tf.zeros([128, 32]), name = 'weight2')
b8 = tf.Variable(tf.zeros([32]), name = 'bias1')
layer8 = tf.matmul(layer7, w8) + b8

# 9                       
w9 = tf.Variable(tf.zeros([32, 10]), name = 'weight2')
b9 = tf.Variable(tf.zeros([10]), name = 'bias1')
hypothesis = tf.nn.softmax(tf.matmul(layer8, w9) + b9)                 # 마지막 output_layer

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(loss)
# model.add(Dense(1, input_dim = 50))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #3. 
    y_train = sess.run(tf.one_hot(y_train, 10))
    y_test = sess.run(tf.one_hot(y_test, 10))
    

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict = {x:x_train, y:y_train})

        if step % 200 ==0:
            print(step, cost_val)

    # 최적의 W와 b가 구해져 있다
    a = sess.run(hypothesis, feed_dict={x:x_test})
    y_pred = sess.run(tf.argmax(a, 1))
    print(a, y_pred )

    # #1. Accuracy - sklearn
    # y_pred = sess.run(tf.one_hot(y_pred, 3))
    # # y_test = sess.run(tf.argmax(y_test, 1))       # y_test(원핫), y_pred(argmax) 의 모양을 같게 만들어 주기 위해서
    # acc = accuracy_score(y_test, y_pred)
    # print('Accuracy :', acc)

    #2. Accuracy - tensorflow
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy : ', sess.run(accuracy, feed_dict={x: x_test, y: y_test}))
