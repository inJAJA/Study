import tensorflow as tf
tf.set_random_seed(777)

x_train = [1, 2, 3]
y_train = [3, 5, 7]
                                                        # 우리가 사용하는 변수와 동일
W = tf.Variable(tf.random_normal([1]), name = 'weight') # 단, Variable사용시 초기화 필수
b = tf.Variable(tf.random_normal([1]), name = 'bias')
                        #_normalization

# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) # 변수 초기화
# print(sess.run(W))                          # [2.2086694]

hypothesis = x_train * W + b                  # model

cost = tf.reduce_mean(tf.square(hypothesis - y_train))   # cost = loss
                                                         # mse

train = tf.train.GradientDescentOptimizer(learning_rate= 0.01).minimize(cost) # cost값 최소화
        # cost를 최소화하기 위해 각 Variable을 천천히 변경하는 optimizer 

with tf.Session() as sess:                        # with을 쓰면 open, close를 안써도 됌 / Session을 계속 사용하기 위해 열어둔다
    sess.run(tf.global_variables_initializer())   # 이 이후로 모든 변수들 초기화
                                     
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b]) # session을 이용해 train 훈련

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
