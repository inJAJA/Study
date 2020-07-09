from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import tensorflow as tf

diabetes = load_diabetes()

x_data = diabetes.data
y_data = diabetes.target

y_data = y_data.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state = 66, train_size = 0.8)

print(x_data.shape)           # (442, 10)
print(y_data.shape)           # (442,)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([10, 1]), name = 'weight')
                                # x의 열의 값과 동일 해야 함(행렬 계산) 

b = tf.Variable(tf.random_normal([1]), name = 'bias') 

hypothesis = tf.matmul(x, w) + b                      # wx + b (행렬 곱)

cost =  tf.reduce_mean(tf.square( hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-5) # 0.00001


train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _= sess.run([cost, hypothesis, train],
                                feed_dict = {x: x_train, y: y_train})

    if step % 20 == 0 :
        print(step, 'cost :',cost_val, '\n 예측값 :', hy_val)

y_pred = sess.run(hypothesis, feed_dict={x:x_test})

r2 = r2_score(y_test, y_pred)
print("R2 :", r2)
