import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [1., 2., 3.]

W = tf.placeholder(tf.float32)          # W 값만 받아 들이겠다

hyporthesis = x * W

cost = tf.reduce_mean(tf.square(hyporthesis - y))

w_history = []                          # keras model.fit의 반환값 history 구현
cost_history = []

with tf.Session() as sess:
    for i in range(-30, 50):    
        curr_w = i * 0.1                # W 설정 : 그림 그릴 간격
        curr_cost = sess.run(cost, feed_dict = {W : curr_w})

        w_history.append(curr_w)        # W의 변동값
        cost_history.append(curr_cost)  # cost의 변동값

plt.plot(w_history, cost_history)       # 가로축 = W / 세로축 = cost
plt.xlabel('W')
plt.ylabel('cost')
plt.show()