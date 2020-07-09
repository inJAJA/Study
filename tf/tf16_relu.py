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
w = tf.Variable(tf.zeros([784, 64]), name = 'weight1')
b = tf.Variable(tf.zeros([64]), name = 'bias1')

layer = tf.nn.relu(tf.matmul(x, w) + b)
# layer = tf.nn.elu(tf.matmul(x, w) + b)
# layer = tf.nn.selu(tf.matmul(x, w) + b)
# layer = tf.nn.sigmoid(tf.matmul(x, w) + b)
# layer = tf.nn.softmax(tf.matmul(x, w) + b)

# dropout
layer = tf.nn.dropout(layer, keep_prob=0.3)


# 2                      
w = tf.Variable(tf.zeros([64, 10]), name = 'weight2')
b = tf.Variable(tf.zeros([10]), name = 'bias1')
hypothesis = tf.nn.softmax(tf.matmul(layer, w) + b)                 # 마지막 output_layer

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)
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
