# tf06_01.py를 카피해서 
# lr을 수정해서 연습
# 0.01 -> 0.1 / 0.001 /1
# epoch가 2000번을 적게 만들어라

import tensorflow as tf
tf.set_random_seed(777)

x = [1, 2, 3]
y = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape = [None])    
y_train = tf.placeholder(tf.float32, shape = [None])
                                                        
W = tf.Variable(tf.random_normal([1]), name = 'weight') 
b = tf.Variable(tf.random_normal([1]), name = 'bias')  
                        #_normalization                 

# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) 
# print(sess.run(W))                          

hypothesis = x_train * W + b                  

cost = tf.reduce_mean(tf.square(hypothesis - y_train))   

train = tf.train.GradientDescentOptimizer(learning_rate= 0.12).minimize(cost) 
         

# with tf.Session() as sess:                        
with tf.compat.v1.Session() as sess:
    # sess.run(tf.global_variables_initializer())          
    sess.run(tf.compat.v1.global_variables_initializer()) 
                                     
    for step in range(1001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict = {x_train:[1, 2, 3], y_train:[3, 5, 7]}) 
        # _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict = {x_train:x, y_train:y}) 


        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

    # predict해보자         
    print("예측 :",sess.run(hypothesis, feed_dict={x_train:[4]}))           
    print("예측 :",sess.run(hypothesis, feed_dict={x_train:[5, 6]}))        
    print("예측 :",sess.run(hypothesis, feed_dict={x_train:[6, 7, 8]}))    
