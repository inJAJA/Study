import tensorflow as tf
tf.set_random_seed(777)

x = [1, 2, 3]
y = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape = [None])    # 현재 shape는 모른다.
y_train = tf.placeholder(tf.float32, shape = [None])
                                                        
W = tf.Variable(tf.random_normal([1]), name = 'weight') # 난수를 주는 이유 
b = tf.Variable(tf.random_normal([1]), name = 'bias')   # : 시작 위치가 달라져도 최적의 값을 찾아가는 것을 보기 위함 
                        #_normalization                 # / 상수를 써도 상관 없다.

# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) 
# print(sess.run(W))                          

hypothesis = x_train * W + b                  

cost = tf.reduce_mean(tf.square(hypothesis - y_train))   

train = tf.train.GradientDescentOptimizer(learning_rate= 0.01).minimize(cost) 
         

# with tf.Session() as sess:                        
with tf.compat.v1.Session() as sess:
    # sess.run(tf.global_variables_initializer())          # 변수에 메모리를 할당하고 초기값을 설정하는 역할
    sess.run(tf.compat.v1.global_variables_initializer())  # 이렇게 쓰는 것을 권한다.
                                     
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict = {x_train:[1, 2, 3], y_train:[3, 5, 7]}) 
        # _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict = {x_train:x, y_train:y}) 


        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

    # predict해보자         
    print("예측 :",sess.run(hypothesis, feed_dict={x_train:[4]}))           # 예측 : [9.000078]
    print("예측 :",sess.run(hypothesis, feed_dict={x_train:[5, 6]}))        # 예측 : [11.000123 13.000169]
    print("예측 :",sess.run(hypothesis, feed_dict={x_train:[6, 7, 8]}))     # 예측 : [13.000169 15.000214 17.000257]
    # 위에서 구한 최적의 W와 b를 구한 것을 사용하여 hypothesis 계산
