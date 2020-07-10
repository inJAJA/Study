import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist          
import numpy as np   

#------------------------data--------------------------
mnist.load_data()                                         

(x_train, y_train), (x_test, y_test) = mnist.load_data() 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255.

# x_train = x_train.reshape(-1, 28*28).astype('float32')/255.
# x_test = x_test.reshape(-1, 28*28).astype('float32')/255.

print(x_train.shape)   # (60000, 28, 28, 1)
print(y_train.shape)   # (60000, 10) 

learning_rate = 1e-3
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size)  # 60000 / 100

x = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])
# x = tf.placeholder(tf.float32, shape = [None, 784])
# x_img = tf.reshape(x, [-1, 28, 28, 1])                  # input_shape

y = tf.placeholder(tf.float32, shape = [None, 10])

keep_prob = tf.placeholder(tf.float32)        # dropout


#-------------------------- model ----------------------------

#1.                                
# Conv2D(32, (3, 3), input_shape= (28, 28, 1))   
w1 = tf.get_variable('w1', shape=[3, 3, 1, 32])  # [kernel_size, kernel_size, color, output]
print(w1)                                        # (3, 3, 1, 32)
                                                 # Conv2D에는 bias 계산이 자동으로 되서 b 따로 명시 안해줘도 됨 
L1 = tf.nn.conv2d(x, w1, strides = [1, 1, 1, 1], padding = 'SAME')  # strides :예제에서는 가운데 두개만 적용됨 (1, 1)
print(L1)                                        # (?, 28, 28, 32)
L1 = tf.nn.selu(L1)                              # conv2d로 연산시킨 결과를 actvation을 통과시킨다.
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding = 'SAME')            # kernel_size = (2, 2), strides = (2, 2)
print(L1)                                        # (?, 14, 14, 32)


#2.                        # ksize / input / output
w2 = tf.get_variable('w2', shape=[3, 3, 32, 64])                        
L2 = tf.nn.conv2d(L1, w2, strides = [1, 1, 1, 1], padding = 'SAME')
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding = 'SAME')


#3.                        # ksize / input / output
w3 = tf.get_variable('w3', shape=[3, 3, 64, 128])                        
L3 = tf.nn.conv2d(L2, w3, strides = [1, 1, 1, 1], padding = 'SAME')
L3 = tf.nn.selu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding = 'SAME')

#4.                        # ksize / input / output
w4 = tf.get_variable('w4', shape=[3, 3, 128, 64])                        
L4 = tf.nn.conv2d(L3, w4, strides = [1, 1, 1, 1], padding = 'SAME')
L4 = tf.nn.selu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding = 'SAME')
print(L4)                                          # (?, 2, 2, 64)
print('=========================================================')

# Flatten
L_flat = tf.reshape(L4, [-1, 2*2*64])

#1. 
w1 = tf.get_variable('w5', shape=[2*2*64, 64],                       
                    initializer=tf.contrib.layers.xavier_initializer())                        
b1 = tf.Variable(tf.random_normal([64]), name = 'bias')
L1 = tf.nn.selu(tf.matmul(L_flat, w1) + b1)

#2. 
w2 = tf.get_variable('w6', shape=[64, 32],                       
                    initializer=tf.contrib.layers.xavier_initializer())                        
b2 = tf.Variable(tf.random_normal([32]), name = 'bias')
L2 = tf.nn.selu(tf.matmul(L1, w2) + b2)

#3. 
w3 = tf.get_variable('w7', shape=[32, 10],                       
                    initializer=tf.contrib.layers.xavier_initializer())                        
b3 = tf.Variable(tf.random_normal([10]), name = 'bias')
hypothesis = tf.nn.softmax(tf.matmul(L2, w3) + b3)             # softmax

#------------------------------- compile --------------------------------

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))   # cross_entropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

#-------------------------------- fit ------------------------------------

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):            # 15
    ave_cost = 0

    for i in range(total_batch):                # 600
        start = i*batch_size                    
        end = start + batch_size                

        batch_xs, batch_ys = x_train[start : end], y_train[start : end]
        
        feed_dict = {x:batch_xs, y:batch_ys, keep_prob:0.9}           # (1 - keep_prob)만큼 dropout한다.
        c, _=sess.run([loss, optimizer], feed_dict=feed_dict) 
        ave_cost += c / total_batch

    print('Epoch :', '%04d'%(epoch+1),
          'loss =', '{:.9f}'.format(ave_cost))

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc :', sess.run(accuracy, feed_dict = {x:x_test, y:y_test, keep_prob:1}))        # Acc : 0.9631
