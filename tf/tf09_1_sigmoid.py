import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]

y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape =[None, 1])
 
w = tf.Variable(tf.random_normal([2, 1]), name = 'weight')
                                # x의 열의 값과 동일 해야 함(행렬 계산) 

b = tf.Variable(tf.random_normal([1]), name = 'bias') 

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)     # sigmoid(wx + b) : (행렬 곱)

# cost =  tf.reduce_mean(tf.square( hypothesis - y))  
cost =  -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))  # sigmoid에 대한 cost
                                                                                # cross-entropy식

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 4.9e-2) # 0.00001
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)                         # 0.5 이상 = 1 / 0.5 이하 = 0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))  
                                  # tf.equal : predicte와 y가 같냐
                        # tf.cast : boolen형 일 때에 True = 1, False = 0
                        #         : 입력 값을 부동소수점 값으로 변경합니다.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _= sess.run([cost, train], feed_dict = {x: x_data, y: y_data})
    
        if step % 10 == 0 :
            print(step, 'cost :',cost_val)
    
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={x:x_data, y:y_data})
    print('\n Hypothesis :', h, '\n Correct (y) :', c, 
          '\n Accuracy :', a)
