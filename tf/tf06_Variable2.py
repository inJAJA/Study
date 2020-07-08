# hypothesis를 구하시오.
# H = Wx + b
# aaa, bbb, ccc 자리에 각 hypothesis를 구하시오.

import tensorflow as tf
tf.set_random_seed(777)

x = [1, 2, 3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = W * x + b

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # 변수 선언
aaa = sess.run(hypothesis)
print('hypothesis :',aaa)                    # hypothesis : [1.3       1.6       1.9000001]
sess.close()                                 # Session()은 메모리를 열어서 작업함으로 작업 후 닫아 주어야 한다.


# InteractiveSession                         # Sessin과 동일 / 사용 방식만 다름
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = hypothesis.eval()
print('hypothesis :',bbb)                    # hypothesis : [1.3       1.6       1.9000001]
sess.close()


sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = hypothesis.eval(session = sess)        # Session에서 .eval 사용법
print('hypothesis :',ccc)                    # hypothesis : [1.3       1.6       1.9000001]
sess.close()