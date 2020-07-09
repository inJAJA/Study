# 다중 분류
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

iris = load_iris()
x_data = iris.data
y_data = iris.target

print(x_data.shape) # (150, 4)
print(y_data.shape) # (150, )

#1. tf.Session() 한번 더 쓰고 하는 방법
# sess = tf.Session()
# y_data = sess.run(tf.one_hot(y_data, 3))
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state = 66, train_size = 0.2)

x = tf.placeholder(tf.float32, shape = [None, 4])
y = tf.placeholder(tf.float32, shape = [None, 3])

W = tf.Variable(tf.random_normal([4, 3]), name = 'weigth' )
b = tf.Variable(tf.random_normal([1, 3]), name = 'bias' )

hypothesis = tf.nn.softmax(tf.matmul(x, W) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #2. train_test_split사용후 원-핫
    # y_train = sess.run(tf.one_hot(y_train, depth = 3)) # one_hot 인코딩
    # y_test = sess.run(tf.one_hot(y_test, depth = 3)) # one_hot 인코딩
    # print(y_train)
    # print(y_test)
    
    #3. 
    y_data = sess.run(tf.one_hot(y_data, 3))
    # print(y_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state = 66, train_size = 0.8)
    

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict = {x:x_train, y:y_train})

        if step % 200 ==0:
            print(step, cost_val)

    # 최적의 W와 b가 구해져 있다
    a = sess.run(hypothesis, feed_dict={x:x_test})
    y_pred = sess.run(tf.argmax(a, 1))
    print(a, y_pred )

    #1. Accuracy - sklearn
    y_pred = sess.run(tf.one_hot(y_pred, 3))
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy :', acc)

    #2. Accuracy - tensorflow
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy : ', sess.run(accuracy, feed_dict={x: x_test, y: y_test}))