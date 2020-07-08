import tensorflow as tf
tf.set_random_seed(777)

x = [1, 2, 3]
y = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape = [None])    
y_train = tf.placeholder(tf.float32, shape = [None])
                                                        
W = tf.Variable(tf.random_normal([1]), name = 'weight') 
b = tf.Variable(tf.random_normal([1]), name = 'bias')  

print(W)                                     # <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>          

W = tf.Variable([0.3], tf.float32)

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # 변수 선언
aaa = sess.run(W)
print('aaa :',aaa)                           # aaa : [0.3]
sess.close()                                 # Session()은 메모리를 열어서 작업함으로 작업 후 닫아 주어야 한다.


# InteractiveSession                         # Sessin과 동일 / 사용 방식만 다름
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = W.eval()
print('bbb :',bbb)                           # bbb : [0.3]
sess.close()


sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session = sess)                 # Session에서 .eval 사용법
print('ccc :',ccc)                           # ccc : [0.3]
sess.close()